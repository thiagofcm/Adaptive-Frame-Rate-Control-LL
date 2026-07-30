"""
KF/eval_kf_quality.py
──────────────────────
Standalone quantitative evaluation of the Kalman filter in envs/lunar_lander_var_fps_kf.py --
no PPO, no adaptive-FPS policy. Runs N_EPISODES episodes at each fixed sensing FPS in the
env's own fps_choices list, using the same frozen NavModel checkpoint KF/debug_kf.py uses as
the low-level landing controller (FPS is pinned per run, matching debug_kf.py's approach --
this is purely about KF quality, not about training an adaptive-FPS policy).

For a given episode index, the SAME seed is reused across every FPS value (only the FPS
differs), so terrain/initial-impulse variability isn't a confound when comparing across FPS.

For each episode/tick, logs:
  - true state (KF/debug_kf.py's get_true_state(), pulled from Box2D body properties)
  - kf_x (the KF's predicted/fused estimate)
  - held (naive "repeat last sample" baseline, reconstructed the same way debug_kf.py does)
  - kf_P (full 6x6, not just the diagonal -- NEES needs the full inverse)
  - kf_nis (non-None only on sampling ticks)
  - ticks-since-last-sample (env's own steps_since_last_obs)

Computes and writes, per FPS and per state dimension (dims: x, vx, y, vy, angle,
angular_velocity):
  - RMSE(KF vs true), RMSE(held vs true), and their ratio
  - NEES (joint 6-dim statistic, not decomposable per-dim -- broadcast across the 6 dim rows
    for a given FPS) averaged over all ticks where P isn't the (reset-only) all-zero matrix
  - NIS (joint, same broadcast treatment) mean/var at sampling ticks only
plus two secondary long-form CSVs for the staleness-indexed curves (RMSE vs. staleness depth,
mean trace(P) vs. staleness depth) that don't fit as single per-FPS-dim summary values.

Usage:
    python -m KF.eval_kf_quality
    python -m KF.eval_kf_quality --n_episodes 50 --wind-power 20 --turbulence-power 2 \
        --vertical-wind-power 20 --sensor-noise-std 0.05
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NTHREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)  # so `import envs....` / `from KF...` resolve regardless of cwd

import argparse
import csv
import numpy as np
import torch
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import envs.lunar_lander_var_fps_kf  # noqa: F401 -- registers "LunarLander_VarFramerate_KF"
from KF.debug_kf import get_true_state, NavModel, STATE_NAMES

NAV_MODEL_PATH = os.path.join(
    REPO_ROOT, "runs/LunarLander_GaussianWind__train__windTrue_20.0_2.0_vert20.0__1__1785251869/model.pt"
)

RUN_SEED = 42
STATE_DIM = 6  # x, vx, y, vy, angle, angular_velocity


# ══════════════════════════════════════════════════════
# Episode runner
# ══════════════════════════════════════════════════════

def run_episode(fixed_fps, wind_cfg, nav_model, seed, max_steps):
    env = gym.make(
        "LunarLander_VarFramerate_KF",
        frame_cost=0.0, budget=1e9,
        **wind_cfg,
    )
    u = env.unwrapped
    u.navigation_model = nav_model
    fps_action = u.fps_choices.index(fixed_fps)

    obs, info = env.reset(seed=seed)

    # "Naive held" = what a system with NO Kalman filter but the SAME noisy sensor
    # would have returned: the last noisy measurement z, not the noiseless true state.
    # At reset, no sensor reading has been taken yet -- the KF is seeded directly from
    # a noiseless _physics_step(0) call (see envs/lunar_lander_var_fps_kf.py's reset()),
    # so tick 0 legitimately has zero held-vs-true error here; this matches kf_x's own
    # tick-0 seeding, not an inconsistency.
    held_state = get_true_state(u).copy()

    true_log, kf_log, held_log, P_log, nis_log, staleness_log = [], [], [], [], [], []

    def record(staleness):
        true_log.append(get_true_state(u))
        kf_log.append(u.kf_x.copy())
        held_log.append(held_state.copy())
        P_log.append(u.kf_P.copy())
        nis_log.append(u.kf_nis if u.kf_nis is not None else np.nan)
        staleness_log.append(staleness)

    record(staleness=0)  # tick 0: reset-seeded state (kf_x == true, P == 0 exactly)

    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < max_steps:
        obs, reward, terminated, truncated, info = env.step(fps_action)
        steps += 1
        if bool(info["frame_consumed"]):
            held_state = u.kf_last_z.copy()
        record(staleness=u.steps_since_last_obs)

    env.close()
    return {
        "true": np.array(true_log),
        "kf": np.array(kf_log),
        "held": np.array(held_log),
        "P": np.array(P_log),
        "nis": np.array(nis_log),
        "staleness": np.array(staleness_log),
    }


def run_fps_batch(fixed_fps, wind_cfg, nav_model, n_episodes, max_steps):
    """Runs n_episodes at a single fixed FPS, concatenating all per-tick logs across
    episodes (episode boundaries don't matter for the aggregate stats below)."""
    all_true, all_kf, all_held, all_P, all_nis, all_stale = [], [], [], [], [], []
    for ep in range(n_episodes):
        seed = RUN_SEED + ep  # same seed reused across FPS values -- see module docstring
        ep_log = run_episode(fixed_fps, wind_cfg, nav_model, seed, max_steps)
        all_true.append(ep_log["true"])
        all_kf.append(ep_log["kf"])
        all_held.append(ep_log["held"])
        all_P.append(ep_log["P"])
        all_nis.append(ep_log["nis"])
        all_stale.append(ep_log["staleness"])
        print(f"  fps={fixed_fps:>3d} | episode {ep + 1}/{n_episodes} | "
              f"{len(ep_log['true'])} ticks")

    return {
        "true": np.concatenate(all_true, axis=0),
        "kf": np.concatenate(all_kf, axis=0),
        "held": np.concatenate(all_held, axis=0),
        "P": np.concatenate(all_P, axis=0),
        "nis": np.concatenate(all_nis, axis=0),
        "staleness": np.concatenate(all_stale, axis=0),
    }


# ══════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════

def compute_nees(err, P):
    """err: (N, 6), P: (N, 6, 6). Returns (N,) NEES values, NaN where P is the exact
    all-zero matrix (only possible at the reset tick, or throughout if Q is exactly 0
    -- e.g. wind_power=vertical_wind_power=turbulence_power=0 -- both cases genuinely
    undefined, not a bug to paper over)."""
    n = err.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        if np.allclose(P[i], 0.0):
            continue
        out[i] = err[i] @ np.linalg.inv(P[i]) @ err[i]
    return out


def compute_fps_dim_summary(fps, batch):
    """Returns a list of dict rows, one per state dimension, for this FPS."""
    err_kf = batch["kf"] - batch["true"]      # (N, 6)
    err_held = batch["held"] - batch["true"]  # (N, 6)

    rmse_kf = np.sqrt(np.mean(err_kf ** 2, axis=0))       # (6,)
    rmse_held = np.sqrt(np.mean(err_held ** 2, axis=0))   # (6,)
    with np.errstate(divide="ignore", invalid="ignore"):
        # rmse_held can be exactly 0 at high sampling FPS (e.g. every-tick sampling ->
        # "held" degenerates to the noiseless true state) -- ratio is legitimately inf
        # there (KF still carries injected sensor noise), not a bug to paper over.
        ratio = rmse_kf / rmse_held

    nees = compute_nees(err_kf, batch["P"])
    nees_mean = np.nanmean(nees)

    nis_valid = batch["nis"][~np.isnan(batch["nis"])]
    nis_mean = np.mean(nis_valid) if len(nis_valid) else np.nan
    nis_var = np.var(nis_valid) if len(nis_valid) else np.nan

    rows = []
    for d in range(STATE_DIM):
        rows.append({
            "fps": fps,
            "dim": STATE_NAMES[d],
            "rmse_kf": rmse_kf[d],
            "rmse_held": rmse_held[d],
            "rmse_ratio_kf_over_held": ratio[d],
            "nees_mean": nees_mean,          # joint stat, broadcast across dim rows
            "nees_expected": STATE_DIM,
            "nis_mean": nis_mean,            # joint stat, broadcast across dim rows
            "nis_var": nis_var,
            "nis_expected": STATE_DIM,
            "n_ticks": batch["true"].shape[0],
        })
    return rows


def compute_staleness_curves(fps, batch):
    """Long-form rows: one per (fps, dim, staleness) for RMSE, and one per
    (fps, staleness) for mean trace(P)."""
    err_kf = batch["kf"] - batch["true"]
    err_held = batch["held"] - batch["true"]
    stale_vals = np.unique(batch["staleness"])
    trace_P = batch["P"].trace(axis1=1, axis2=2)  # (N,)

    rmse_rows = []
    traceP_rows = []
    for s in stale_vals:
        mask = batch["staleness"] == s
        n = int(mask.sum())
        for d in range(STATE_DIM):
            rmse_rows.append({
                "fps": fps,
                "dim": STATE_NAMES[d],
                "staleness": int(s),
                "rmse_kf": np.sqrt(np.mean(err_kf[mask, d] ** 2)),
                "rmse_held": np.sqrt(np.mean(err_held[mask, d] ** 2)),
                "n_samples": n,
            })
        traceP_rows.append({
            "fps": fps,
            "staleness": int(s),
            "mean_trace_P": float(np.mean(trace_P[mask])),
            "n_samples": n,
        })
    return rmse_rows, traceP_rows


# ══════════════════════════════════════════════════════
# CSV writers
# ══════════════════════════════════════════════════════

def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"CSV saved -> {path}")


# ══════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════

def plot_rmse_ratio_heatmap(summary_rows, fps_list, out_dir):
    grid = np.zeros((STATE_DIM, len(fps_list)))
    for r in summary_rows:
        d = STATE_NAMES.index(r["dim"])
        j = fps_list.index(r["fps"])
        grid[d, j] = r["rmse_ratio_kf_over_held"]

    fig, ax = plt.subplots(figsize=(1.6 * len(fps_list) + 2, 5))
    im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=0.0, vmax=2.0)
    ax.set_xticks(range(len(fps_list)))
    ax.set_xticklabels(fps_list)
    ax.set_yticks(range(STATE_DIM))
    ax.set_yticklabels(STATE_NAMES)
    ax.set_xlabel("fixed FPS")
    ax.set_title("RMSE ratio (KF / naive-held) -- <1 means KF beats naive hold")
    for d in range(STATE_DIM):
        for j in range(len(fps_list)):
            ax.text(j, d, f"{grid[d, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="RMSE ratio (KF/held)")
    fig.tight_layout()
    path = os.path.join(out_dir, "rmse_ratio_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_rmse_vs_staleness(rmse_curve_rows, fps_list, out_dir):
    fig, axes = plt.subplots(STATE_DIM, 1, figsize=(10, 22), sharex=False)
    colors = plt.cm.viridis(np.linspace(0, 1, len(fps_list)))
    for d, name in enumerate(STATE_NAMES):
        ax = axes[d]
        for c, fps in zip(colors, fps_list):
            rows = [r for r in rmse_curve_rows if r["dim"] == name and r["fps"] == fps]
            rows.sort(key=lambda r: r["staleness"])
            s = [r["staleness"] for r in rows]
            ax.plot(s, [r["rmse_kf"] for r in rows], color=c, linestyle="-", label=f"KF fps={fps}")
            ax.plot(s, [r["rmse_held"] for r in rows], color=c, linestyle=":", label=f"held fps={fps}")
        ax.set_ylabel(name)
        if d == 0:
            ax.legend(fontsize=7, ncol=len(fps_list))
    axes[-1].set_xlabel("ticks since last sample (staleness depth)")
    fig.suptitle("RMSE vs. staleness depth (solid=KF, dotted=naive held)")
    fig.tight_layout()
    path = os.path.join(out_dir, "rmse_vs_staleness.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_traceP_vs_staleness(traceP_curve_rows, fps_list, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(fps_list)))
    for c, fps in zip(colors, fps_list):
        rows = [r for r in traceP_curve_rows if r["fps"] == fps]
        rows.sort(key=lambda r: r["staleness"])
        ax.plot([r["staleness"] for r in rows], [r["mean_trace_P"] for r in rows],
                color=c, marker="o", markersize=3, label=f"fps={fps}")
    ax.set_xlabel("ticks since last sample (staleness depth)")
    ax.set_ylabel("mean trace(kf_P)")
    ax.set_title("KF uncertainty vs. staleness depth, per FPS")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "traceP_vs_staleness.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_consistency(summary_rows, fps_list, out_dir):
    """NEES and NIS per FPS (one value per FPS -- pull from any of that FPS's 6
    broadcast-identical dim rows), each against its expected value (state dim, 6)."""
    nees_by_fps = {r["fps"]: r["nees_mean"] for r in summary_rows}
    nis_by_fps = {r["fps"]: r["nis_mean"] for r in summary_rows}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.bar([str(f) for f in fps_list], [nees_by_fps[f] for f in fps_list], color="tab:blue")
    ax1.axhline(STATE_DIM, color="black", linestyle="--", label=f"expected = {STATE_DIM}")
    ax1.set_xlabel("fixed FPS")
    ax1.set_ylabel("mean NEES")
    ax1.set_title("NEES per FPS (all ticks)")
    ax1.legend(fontsize=8)

    ax2.bar([str(f) for f in fps_list], [nis_by_fps[f] for f in fps_list], color="tab:orange")
    ax2.axhline(STATE_DIM, color="black", linestyle="--", label=f"expected = {STATE_DIM}")
    ax2.set_xlabel("fixed FPS")
    ax2.set_ylabel("mean NIS")
    ax2.set_title("NIS per FPS (sampling ticks only)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, "consistency_nees_nis.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--nav_model_path", type=str, default=NAV_MODEL_PATH)
    parser.add_argument("--enable-wind", dest="enable_wind", action="store_true", default=True)
    parser.add_argument("--no-enable-wind", dest="enable_wind", action="store_false")
    parser.add_argument("--wind-power", type=float, default=20.0)
    parser.add_argument("--turbulence-power", type=float, default=2.0)
    parser.add_argument("--vertical-wind-power", type=float, default=20.0)
    parser.add_argument("--sensor-noise-std", type=float, default=0.05)
    parser.add_argument("--output_dir", type=str, default=os.path.join(REPO_ROOT, "KF", "kf_quality_results"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    wind_cfg = dict(
        enable_wind=args.enable_wind, wind_power=args.wind_power,
        turbulence_power=args.turbulence_power, vertical_wind_power=args.vertical_wind_power,
        sensor_noise_std=args.sensor_noise_std,
    )
    print(f"Wind config: {wind_cfg}")

    device = torch.device("cpu")
    nav_model = NavModel(args.nav_model_path, device)
    print(f"Nav model loaded: {args.nav_model_path}")

    # Read fps_choices straight from the env so this stays in sync automatically.
    probe_env = gym.make("LunarLander_VarFramerate_KF")
    fps_list = list(probe_env.unwrapped.fps_choices)
    probe_env.close()
    print(f"Evaluating fixed FPS values: {fps_list} x {args.n_episodes} episodes each")

    summary_rows = []
    rmse_curve_rows = []
    traceP_curve_rows = []

    for fixed_fps in fps_list:
        print(f"\n=== fixed FPS = {fixed_fps} ===")
        batch = run_fps_batch(fixed_fps, wind_cfg, nav_model, args.n_episodes, args.max_steps)
        summary_rows.extend(compute_fps_dim_summary(fixed_fps, batch))
        rmse_rows, traceP_rows = compute_staleness_curves(fixed_fps, batch)
        rmse_curve_rows.extend(rmse_rows)
        traceP_curve_rows.extend(traceP_rows)

    # --- CSVs ---
    write_csv(
        os.path.join(args.output_dir, "summary_per_fps_dim.csv"),
        summary_rows,
        ["fps", "dim", "rmse_kf", "rmse_held", "rmse_ratio_kf_over_held",
         "nees_mean", "nees_expected", "nis_mean", "nis_var", "nis_expected", "n_ticks"],
    )
    write_csv(
        os.path.join(args.output_dir, "staleness_rmse.csv"),
        rmse_curve_rows,
        ["fps", "dim", "staleness", "rmse_kf", "rmse_held", "n_samples"],
    )
    write_csv(
        os.path.join(args.output_dir, "staleness_traceP.csv"),
        traceP_curve_rows,
        ["fps", "staleness", "mean_trace_P", "n_samples"],
    )

    # --- Plots ---
    plot_rmse_ratio_heatmap(summary_rows, fps_list, args.output_dir)
    plot_rmse_vs_staleness(rmse_curve_rows, fps_list, args.output_dir)
    plot_traceP_vs_staleness(traceP_curve_rows, fps_list, args.output_dir)
    plot_consistency(summary_rows, fps_list, args.output_dir)

    # --- Console summary ---
    print("\n===== Summary (RMSE ratio KF/held, <1 = KF better) =====")
    for fps in fps_list:
        print(f"\nfps={fps}:")
        for r in summary_rows:
            if r["fps"] == fps:
                print(f"  {r['dim']:20s} rmse_kf={r['rmse_kf']:.4f}  rmse_held={r['rmse_held']:.4f}  "
                      f"ratio={r['rmse_ratio_kf_over_held']:.3f}")
        fps_rows = [r for r in summary_rows if r["fps"] == fps]
        print(f"  NEES mean = {fps_rows[0]['nees_mean']:.2f} (expected {STATE_DIM})")
        print(f"  NIS  mean = {fps_rows[0]['nis_mean']:.2f}, var = {fps_rows[0]['nis_var']:.2f} (expected mean {STATE_DIM})")

    print(f"\nAll outputs written to: {args.output_dir}")
