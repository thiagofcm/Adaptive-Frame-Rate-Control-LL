#!/usr/bin/env python3
"""Sweep runner for experiments/var_fps/train.py (LunarLander_VarFramerate).

Reads a sweep config (default: config/sweep.yaml) containing every base
hyperparameter plus a `sweep:` section mapping parameter names to lists of values.
Builds the Cartesian product of the swept parameters, materializes one full config
file per combination, and launches train.py (this directory) once per combination --
each pinned to its own isolated CPU core (via taskset) and round-robined across
available GPUs.
"""
import argparse
import itertools
import os
import subprocess
import sys
import time
from datetime import datetime

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train.py")
DEFAULT_SWEEP_CONFIG = os.path.join(REPO_ROOT, "config", "sweep.yaml")
# train.py subprocesses run with this as cwd -- matches config/var_fps.yaml's
# nav_model_path (repo-root-relative) and the CarRacing reference's own convention, so
# `runs/{run_name}` output lands in <repo_root>/runs/.
LAUNCH_CWD = REPO_ROOT


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default=DEFAULT_SWEEP_CONFIG,
                    help="yaml file with base hyperparameters + a 'sweep:' section of param -> list of values")
    p.add_argument("--total-timesteps", type=int, default=None,
                    help="override total_timesteps for every run in the sweep (default: whatever the sweep config has)")
    p.add_argument("--gpus", type=int, nargs="+", default=None,
                    help="GPU ids to round-robin runs across; omit to auto-detect, pass -1 for CPU-only")
    p.add_argument("--cpus", type=int, nargs="+", default=None,
                    help="flat pool of CPU core ids to partition across runs, via taskset -- 1 core "
                         "per combo normally, or num_envs+1 per combo when that combo's config has "
                         "async_envs=true (must supply the full pool sized for whichever combos need "
                         "more than 1 -- see compute_cores_per_combo); omit to auto-detect cores not "
                         "already claimed by another process (see detect_free_cpus)")
    mode_group = p.add_mutually_exclusive_group()
    mode_group.add_argument("--sequential", action="store_true",
                    help="run one combination at a time instead of launching them all in parallel")
    mode_group.add_argument("--max-parallel", type=int, default=None,
                    help="cap on how many training runs are alive simultaneously; when one finishes, "
                         "the next pending combination launches immediately, reusing its CPU cores "
                         "(mutually exclusive with --sequential; only needs cores for the max "
                         "concurrently running jobs, not every combination in the sweep upfront)")
    p.add_argument("--extra", type=str, default="",
                    help="extra CLI args forwarded verbatim to train.py")
    args = p.parse_args()
    if args.max_parallel is not None and args.max_parallel < 1:
        p.error("--max-parallel must be >= 1")
    return args


def load_sweep_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    sweep_params = cfg.pop("sweep", None)
    if not sweep_params:
        raise ValueError(f"No non-empty 'sweep:' section found in {path}")
    return cfg, sweep_params


def sweep_combinations(sweep_params):
    keys = list(sweep_params.keys())
    value_lists = [sweep_params[k] for k in keys]
    for values in itertools.product(*value_lists):
        yield dict(zip(keys, values))


def combo_tag(overrides):
    return "_".join(f"{k}{v}" for k, v in overrides.items())


def detect_gpus():
    # CPU-only by default: the Agent is a tiny MLP+LSTM (no GPU benefit at this size),
    # and NavModel does batch-1 MLP inference once per physics tick, where GPU kernel-
    # launch/host<->device-copy overhead dominates over the actual compute -- CPU wins
    # both. SyncVectorEnv also runs every env sequentially regardless of device, so
    # GPU's real advantage (batching across envs) is never exploited here either way.
    # Pass --gpus explicitly (e.g. --gpus 0 1) to opt back into GPU per combo.
    return [-1]


def detect_free_cpus(n_needed):
    """Pick the n_needed *least contended* core ids, ranked by how many other
    processes are already exclusively pinned (via taskset -c <single core>) to each
    one -- rather than always handing out a fixed 0..n_needed-1.

    Without this, two concurrent sweep launches independently taskset their combos
    onto cores 0, 1, 2, ... -- CPU affinity restricts a process to a core, it doesn't
    reserve it, so two processes pinned to the same single core just get time-sliced
    ~50/50 by the scheduler.

    On a heavily shared machine, "fully idle core" may not exist at all. Ranking by
    contention still helps in that case: it picks whichever cores happen to be least
    loaded right now instead of blindly colliding with this project's *own* previous
    launch on a fixed range.
    """
    try:
        import psutil
    except ImportError:
        print("[sweep] psutil not available -- falling back to cores 0..N-1 with no "
              "collision detection. Install psutil, or pass --cpus explicitly if "
              "another sweep might already be running.")
        return list(range(n_needed))

    all_cores = sorted(os.sched_getaffinity(0))
    counts = {c: 0 for c in all_cores}
    owners = {c: [] for c in all_cores}
    for proc in psutil.process_iter(["pid"]):
        try:
            affinity = proc.cpu_affinity()
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            continue
        # Only single-core affinity counts as "claimed" -- that's the taskset -c
        # pattern this script (and any other sweep launch) uses; a process with the
        # full default affinity isn't deliberately pinned to anything specific.
        if len(affinity) == 1 and affinity[0] in counts:
            counts[affinity[0]] += 1
            owners[affinity[0]].append(proc.pid)

    ranked = sorted(all_cores, key=lambda c: counts[c])
    chosen = ranked[:n_needed]

    if any(counts[c] > 0 for c in chosen):
        still_shared = {c: owners[c] for c in chosen if counts[c] > 0}
        print(f"[sweep] No fully idle cores available -- picked the {n_needed} "
              f"least-contended ones instead. Still-shared: {still_shared}")
    else:
        print(f"[sweep] Found {n_needed} fully idle cores: {chosen}")
    return chosen


def compute_cores_per_combo(base_cfg, overrides):
    """How many CPU cores this combo needs: 1 normally, or num_envs+1 (one core per
    AsyncVectorEnv worker subprocess, +1 for the main process) when async_envs is on.

    Async workers are forked from the main process and inherit its CPU affinity mask
    at fork time -- taskset-ing the whole combo to a single core (the old behavior)
    silently pins every worker to that same one core too, so Async gets zero real
    parallelism and just pays IPC overhead on top of Sync-equivalent throughput.
    """
    merged = {**base_cfg, **overrides}
    if merged.get("async_envs", False):
        return int(merged.get("num_envs", 1)) + 1
    return 1


def launch(run_cfg_path, cpu_block, gpu, args, log_dir, tag):
    cpu_str = ",".join(str(c) for c in cpu_block)
    cmd = ["taskset", "-c", cpu_str, sys.executable, "-u", TRAIN_SCRIPT, "--config", run_cfg_path]
    if gpu == -1:
        cmd += ["--no-cuda"]
    if args.extra:
        cmd += args.extra.split()

    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    if gpu >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    log_path = os.path.join(log_dir, f"{tag}.log")
    log_file = open(log_path, "w")
    print(f"[sweep] launching {tag} on cpus={cpu_str} gpu={gpu} -> {log_path}")
    proc = subprocess.Popen(cmd, cwd=LAUNCH_CWD, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    return proc, log_file


def run_max_parallel(combos, cores_needed, cpus, gpus, base_cfg, args, cfg_dir, log_dir, max_parallel):
    """Keep at most max_parallel training runs alive simultaneously. CPU cores are
    handed out from a shared pool (sized in main() for the max concurrently running
    jobs, not every combo in the sweep) and returned to the pool as soon as a run
    finishes, so the next pending combination can immediately reuse them."""
    free_cpus = list(cpus)
    pending = list(enumerate(combos))
    running = []  # list of dicts: tag, proc, log_file, cpu_block

    def launch_one():
        i, overrides = pending.pop(0)
        merged = {**base_cfg, **overrides}
        if args.total_timesteps is not None:
            merged["total_timesteps"] = args.total_timesteps

        tag = combo_tag(overrides)
        run_cfg_path = os.path.join(cfg_dir, f"{tag}.yaml")
        with open(run_cfg_path, "w") as f:
            yaml.safe_dump(merged, f)

        n = cores_needed[i]
        cpu_block, free_cpus[:] = free_cpus[:n], free_cpus[n:]
        gpu = gpus[i % len(gpus)]
        proc, log_file = launch(run_cfg_path, cpu_block, gpu, args, log_dir, tag)
        running.append({"tag": tag, "proc": proc, "log_file": log_file, "cpu_block": cpu_block})
        time.sleep(1)  # stagger run_name timestamps so directories can't collide

    while pending and len(running) < max_parallel:
        launch_one()

    while running:
        time.sleep(2)
        for entry in list(running):
            if entry["proc"].poll() is not None:
                entry["log_file"].close()
                print(f"[sweep] {entry['tag']} finished (exit={entry['proc'].returncode})")
                free_cpus.extend(entry["cpu_block"])
                running.remove(entry)
                if pending:
                    launch_one()


def main():
    args = parse_args()
    base_cfg, sweep_params = load_sweep_config(args.config)
    combos = list(sweep_combinations(sweep_params))

    # Cores needed per combo: 1 normally, num_envs+1 when a combo's merged config has
    # async_envs=True (see compute_cores_per_combo) -- Async workers need real distinct
    # cores, not a share of one, or they get zero parallelism benefit.
    cores_needed = [compute_cores_per_combo(base_cfg, overrides) for overrides in combos]

    if args.max_parallel is not None:
        # Only the max concurrently running jobs need distinct cores -- sized
        # conservatively (max_parallel * the largest single per-combo requirement) so
        # any subset of max_parallel combos can run together, regardless of which
        # ones happen to land in the same window.
        max_parallel = min(args.max_parallel, len(combos)) if combos else 0
        total_cores_needed = max_parallel * max(cores_needed) if cores_needed else 0
    else:
        total_cores_needed = sum(cores_needed)

    gpus = args.gpus if args.gpus is not None else detect_gpus()
    cpus = args.cpus if args.cpus is not None else detect_free_cpus(total_cores_needed)
    if len(cpus) < total_cores_needed:
        if args.max_parallel is not None:
            raise ValueError(f"--max-parallel {max_parallel} needs {total_cores_needed} distinct CPU "
                              f"cores (max_parallel * the largest per-combo requirement), only got "
                              f"{len(cpus)} via --cpus; omit --cpus to auto-assign")
        raise ValueError(f"{len(combos)} combinations need {total_cores_needed} distinct CPU cores total "
                          f"(sum of per-combo requirements -- higher than {len(combos)} when any combo "
                          f"uses async_envs), only got {len(cpus)} via --cpus; omit --cpus to auto-assign")

    date_str = datetime.now().strftime("%d-%m-%H-%M-%S")
    run_root = os.path.join(SCRIPT_DIR, "runs", f"sweep_{date_str}")
    cfg_dir = os.path.join(run_root, "configs")
    log_dir = os.path.join(run_root, "logs")
    os.makedirs(cfg_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if args.max_parallel is not None:
        run_max_parallel(combos, cores_needed, cpus, gpus, base_cfg, args, cfg_dir, log_dir, max_parallel)
        print(f"[sweep] all runs complete. configs -> {cfg_dir}, logs -> {log_dir}")
        print("[sweep] compare with: tensorboard --logdir runs")
        return

    # Partition the flat core pool into one contiguous-in-order block per combo, sized
    # to that combo's own requirement.
    cpu_blocks = []
    _idx = 0
    for n in cores_needed:
        cpu_blocks.append(cpus[_idx:_idx + n])
        _idx += n

    procs = []
    for i, overrides in enumerate(combos):
        merged = {**base_cfg, **overrides}
        if args.total_timesteps is not None:
            merged["total_timesteps"] = args.total_timesteps

        tag = combo_tag(overrides)
        run_cfg_path = os.path.join(cfg_dir, f"{tag}.yaml")
        with open(run_cfg_path, "w") as f:
            yaml.safe_dump(merged, f)

        cpu_block = cpu_blocks[i]
        gpu = gpus[i % len(gpus)]
        proc, log_file = launch(run_cfg_path, cpu_block, gpu, args, log_dir, tag)
        procs.append((tag, proc, log_file))
        time.sleep(1)  # stagger run_name timestamps so directories can't collide
        if args.sequential:
            proc.wait()
            log_file.close()
            print(f"[sweep] {tag} finished (exit={proc.returncode})")

    if not args.sequential:
        for tag, proc, log_file in procs:
            proc.wait()
            log_file.close()
            print(f"[sweep] {tag} finished (exit={proc.returncode})")

    print(f"[sweep] all runs complete. configs -> {cfg_dir}, logs -> {log_dir}")
    print("[sweep] compare with: tensorboard --logdir runs")


if __name__ == "__main__":
    main()
