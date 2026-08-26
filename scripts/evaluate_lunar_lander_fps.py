"""
Stage 1 Lunar Lander evaluation pipeline.

Mirrors the directory philosophy of the F1TENTH AdaptiveFPS evaluator
(AdaptiveFPS/scripts/evaluate_adaptive_fps.py): one shared evaluation loop for both
adaptive and fixed sensing-rate policies, one self-contained directory per episode
(steps.csv + trajectory .npy), and one episodes.csv at the policy-run level.

Uses the existing, unmodified LunarLander_VarFramerate environment
(envs/lunar_lander_var_fps.py) and the frozen navigation controller / adaptive-policy
loading code already established in experiments/var_fps/eval.py. This script does not
change env behavior, reward, or the success definition -- it only evaluates and logs.

Mode selection follows the F1TENTH AdaptiveFPS convention: there is no explicit --mode
flag -- passing --fixed-fps runs the fixed-sensing-rate baseline, and omitting it runs
the adaptive sensing policy (which then requires --model).

Run from the repo root with PYTHONPATH=<repo root>, matching every other script here:

    PYTHONPATH=. python scripts/evaluate_lunar_lander_fps.py \\
        --model <path/to/model.pt> \\
        --frame-cost 2.0 --budget 25 --episodes 100

    PYTHONPATH=. python scripts/evaluate_lunar_lander_fps.py \\
        --fixed-fps 10 \\
        --frame-cost 2.0 --budget 25 --episodes 100 --output-name fixed_10Hz
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NTHREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import argparse
import atexit
import csv
import glob
import json
from datetime import datetime

import numpy as np
import gymnasium as gym
import torch.nn as nn
from torch.distributions.categorical import Categorical
from gymnasium.wrappers import TimeLimit

import envs.lunar_lander_var_fps  # noqa: F401  (registers LunarLander_VarFramerate)

# =========================
# Constants (matching experiments/var_fps/eval.py)
# =========================
OBS_DIM = 11
ACTION_SPACE_LENGTH = 5
RUN_SEED = 42

FPS_TO_ACTION = {1: 0, 5: 1, 10: 2, 25: 3, 50: 4}
FPS_CHOICES = [1, 5, 10, 25, 50]
LSTM_HIDDEN_SIZE = 64


NAV_MODEL_PATH_DEFAULT = "experiments/navigation/runs/LunarLander-v3__ppo__1__1779191150/model.pt"

EVAL_ROOT = "eval/"

STEP_CSV_FIELDS = [
    "step",
    "x", "y", "vx", "vy",
    "angle", "angular_velocity",
    "left_leg_contact", "right_leg_contact",
    "navigation_action",
    "current_fps",
    "fresh_observation",
    "n_fresh_observations",
    "n_fresh_observations_before_touchdown",
    "n_fresh_observations_after_touchdown",
    "instant_nav_reward",
    "cumulative_nav_reward",
    "instant_adaptive_reward",
    "cumulative_adaptive_reward",
    "instant_frame_penalty",
    "terminated",
    "truncated",
    "vy_at_touchdown",
    "success",
]

EPISODE_CSV_FIELDS = [
    "episode_index", "seed", "episode_length",
    "adaptive_reward", "nav_reward",
    "success", "landed_in_flags",
    "outside_flags_after_landing",
    "went_up_after", "exceed_vy_vel",
    "touchdown_vy",
    "n_fresh_observations",
    "n_fresh_observations_before_touchdown",
    "n_fresh_observations_after_touchdown",
    "fresh_observation_ratio",
    "fresh_observation_ratio_b4_td",
    "mean_fps",
]

TRAJECTORY_NPY_COLUMNS = 5  # [x, y, vx, vy, current_fps]

SUMMARY_CSV_NAME = "summary.csv"
SUMMARY_CSV_FIELDS = [
    "model", "mean_fps", "n_episodes", "success_rate",
    "mean_episode_length", "std_episode_length",
    "mean_total_reward", "std_total_reward",
    "mean_touchdown_vy", "std_touchdown_vy",

    "mean_n_fresh_observations",
    "std_n_fresh_observations",

    "mean_n_fresh_observations_before_touchdown",
    "std_n_fresh_observations_before_touchdown",

    "mean_n_fresh_observations_after_touchdown",
    "std_n_fresh_observations_after_touchdown",
]


# ══════════════════════════════════════════════════════
# Model classes -- copied verbatim from experiments/var_fps/eval.py so this evaluator
# preserves the exact frozen-controller and adaptive-sensing-policy architectures.
# ══════════════════════════════════════════════════════

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class NavAgent(nn.Module):
    def __init__(self, obs_dim=8, n_actions=4):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )


class NavModel:
    """Frozen base landing/navigation controller."""

    def __init__(self, model_path, device):
        self.device = device
        checkpoint = torch.load(model_path, map_location=device)
        self.agent = NavAgent().to(device)
        if "model_state_dict" in checkpoint:
            self.agent.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.agent.load_state_dict(checkpoint)
        self.agent.eval()

    def predict(self, obs, deterministic=True):
        obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = torch.argmax(self.agent.actor(obs_tensor), dim=-1)
        return action.cpu().numpy()[0], None


class AgentEval(nn.Module):
    """Adaptive sensing-rate (var-FPS) policy: MLP + LSTM actor-critic."""

    def __init__(self, obs_dim, n_actions, lstm_hidden_size=64):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
        )
        self.lstm = nn.LSTM(64, lstm_hidden_size)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(lstm_hidden_size, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(lstm_hidden_size, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x)
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden.append(h)
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def predict(self, obs, lstm_state, done, deterministic=True):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        done_tensor = torch.FloatTensor([float(done)])
        with torch.no_grad():
            hidden, lstm_state = self.get_states(obs_tensor, lstm_state, done_tensor)
            if deterministic:
                action = torch.argmax(self.actor(hidden), dim=-1)
            else:
                action = Categorical(logits=self.actor(hidden)).sample()
        return action.cpu().numpy()[0], lstm_state


# =========================
# Evaluate one episode -- shared by both adaptive and fixed sensing modes.
# Direct equivalent of AdaptiveFPS's run_lap().
# =========================

def run_episode(env, fixed_fps, episode_dir, model, seed):
    """Run one full episode, write that episode's steps.csv + trajectory .npy, and
    return one summary dict (an episodes.csv row)."""

    observation, info= env.reset(seed=seed)

    lstm_state = (
        torch.zeros(1, 1, LSTM_HIDDEN_SIZE),
        torch.zeros(1, 1, LSTM_HIDDEN_SIZE),
    )

    step = 0
    cumulative_adaptive_reward = 0.0
    cumulative_nav_reward = 0.0
    n_fresh_observations = 1
    n_fresh_observations_before_touchdown = 1
    n_fresh_observations_after_touchdown = 0
    touchdown_step = None
    fps_trace = []
    step_rows = []
    trajectory_rows = []
    done = False
    touchdown_flag = False
    touchdown_vy = None
    landed_in_flags = False
    outside_flags_after_landing = False
    went_up_after = False
    exceed_vy_vel = False   
    prev_leg1 = False
    prev_leg2 = False
    terminated, truncated = False, False

    while not (terminated or truncated):

        if fixed_fps is not None:
            action = FPS_TO_ACTION[fixed_fps]
        else:
            action, lstm_state = model.predict(observation, lstm_state, done, deterministic=True)
        # ------------------------------------------------------

        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

        instant_adaptive_reward = reward
        instant_nav_reward = info["nav_reward"]
        cumulative_adaptive_reward += instant_adaptive_reward
        cumulative_nav_reward += instant_nav_reward

        true_observation = env.unwrapped.current_obs
        navigation_action = info["navigation_action"]

        if info["frame_consumed"]:
            n_fresh_observations += 1
            fresh_observation = True

            if not touchdown_flag:
                n_fresh_observations_before_touchdown += 1
            else:
                n_fresh_observations_after_touchdown += 1

        else:
            fresh_observation = False

        fps_trace.append(info["current_fps"])
        
        # Touchdown / success tracking
        leg1 = bool(true_observation[6])
        leg2 = bool(true_observation[7])
        leg_contact = leg1 or leg2
        vy_at_touchdown = np.nan

        if leg_contact and not touchdown_flag:
            touchdown_vy = abs(true_observation[3])
            touchdown_flag = True
            touchdown_step = step
            landed_in_flags = -0.2 < true_observation[0] < 0.2
            exceed_vy_vel = touchdown_vy > 0.5
            vy_at_touchdown = touchdown_vy

        if touchdown_flag and leg_contact and not (-0.2 < true_observation[0] < 0.2):
            outside_flags_after_landing = True

        both_grounded_prev = prev_leg1 and prev_leg2
        both_off_ground = not leg1 and not leg2
        if touchdown_flag and both_grounded_prev and both_off_ground:
            went_up_after = True

        prev_leg1, prev_leg2 = leg1, leg2

        
        successful = (
            landed_in_flags and not outside_flags_after_landing
            and not went_up_after and not exceed_vy_vel
        )

        step_rows.append({
            "step": step,
            "x": true_observation[0],
            "y": true_observation[1],
            "vx": true_observation[2], 
            "vy": true_observation[3],
            "angle": true_observation[4],
            "angular_velocity": true_observation[5],
            "left_leg_contact": leg1,
            "right_leg_contact": leg2,
            "navigation_action": int(navigation_action),
            "current_fps": info["current_fps"],
            "fresh_observation": fresh_observation,
            "n_fresh_observations": n_fresh_observations,
            "n_fresh_observations_before_touchdown":n_fresh_observations_before_touchdown,
            "n_fresh_observations_after_touchdown":n_fresh_observations_after_touchdown,
            "instant_nav_reward": instant_nav_reward,
            "cumulative_nav_reward": cumulative_nav_reward,
            "instant_adaptive_reward": instant_adaptive_reward,
            "cumulative_adaptive_reward": cumulative_adaptive_reward,
            "instant_frame_penalty": info["frame_penalty"],
            "terminated": terminated, "truncated": truncated,
            "vy_at_touchdown": vy_at_touchdown,
            "success": successful,
        })
        trajectory_rows.append([
            true_observation[0], true_observation[1], true_observation[2], true_observation[3], info["current_fps"],
        ])

    os.makedirs(episode_dir, exist_ok=True)
    with open(f"{episode_dir}/steps.csv", 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=STEP_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(step_rows)
    np.save(f"{episode_dir}/trajectory.npy", np.array(trajectory_rows, dtype=np.float64))

    return {
        "seed": seed,
        "episode_length": step,
        "adaptive_reward": cumulative_adaptive_reward,
        "nav_reward": cumulative_nav_reward,
        "success": successful,
        "landed_in_flags": landed_in_flags,
        "outside_flags_after_landing": outside_flags_after_landing,
        "went_up_after": went_up_after,
        "exceed_vy_vel": exceed_vy_vel,
        "touchdown_vy": touchdown_vy if touchdown_vy is not None else np.nan,
        "n_fresh_observations": n_fresh_observations,
        "n_fresh_observations_before_touchdown":n_fresh_observations_before_touchdown,
        "n_fresh_observations_after_touchdown":n_fresh_observations_after_touchdown,
        "fresh_observation_ratio": n_fresh_observations / step if step else 0.0,
        "fresh_observation_ratio_b4_td": n_fresh_observations_before_touchdown /  touchdown_step if touchdown_step is not None and touchdown_step > 0 else 0.0,
        "mean_fps": float(np.mean(fps_trace)) if fps_trace else "",
    }

def summarize_results():
    """Read every */episodes.csv under eval/lunar_lander/ and combine them into one
    comparison table: one row per policy run (run_name = the directory containing that
    run's episodes.csv), with mean/std stats computed across every evaluated episode for
    that run.

    Registered to run on interpreter exit (see main()) so it always reflects the latest
    accumulated results, whether the run finished normally or was interrupted. This is a
    full rebuild from whatever episodes.csv files currently exist on disk, not an
    incremental update -- ported from AdaptiveFPS's evaluate_adaptive_fps.py, adapted to
    Lunar Lander's single-root eval/lunar_lander/ layout (no per-map subdirectory) and
    field set.
    """
    episode_csv_paths = sorted(glob.glob(f"{EVAL_ROOT}/*/episodes.csv"))
    if not episode_csv_paths:
        return

    summary_rows = []
    for path in episode_csv_paths:
        with open(path, "r", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue

        model = os.path.basename(os.path.dirname(path))

        episode_lengths = np.array([float(r["episode_length"]) for r in rows])
        successes = np.array([r["success"] == "True" for r in rows])
        total_rewards = np.array([float(r["adaptive_reward"]) for r in rows])
        n_fresh = np.array([float(r["n_fresh_observations"]) for r in rows])
        n_fresh_before_touchdown = np.array([float(r["n_fresh_observations_before_touchdown"])for r in rows])
        n_fresh_after_touchdown = np.array([float(r["n_fresh_observations_after_touchdown"])for r in rows])
        # mean of every episode's own mean_fps, not just rows[0]'s -- fixed-FPS runs
        # have an identical value every episode, but an adaptive policy's mean_fps
        # varies episode to episode.
        mean_fps_per_episode = np.array([float(r["mean_fps"]) for r in rows])

        touchdown_vy = np.array([
            float(r["touchdown_vy"]) if r["touchdown_vy"] != "" else np.nan
            for r in rows
        ])
        valid_touchdown_vy = touchdown_vy[~np.isnan(touchdown_vy)]
        if len(valid_touchdown_vy) > 0:
            mean_touchdown_vy = float(valid_touchdown_vy.mean())
            std_touchdown_vy = float(valid_touchdown_vy.std())
        else:
            # No episode in this run ever touched down -- report NaN rather than
            # fabricating a 0, and avoid np.nanmean's "Mean of empty slice" warning.
            mean_touchdown_vy = float("nan")
            std_touchdown_vy = float("nan")

        summary_rows.append({
            "model": model,
            "mean_fps": float(mean_fps_per_episode.mean()),
            "n_episodes": len(rows),
            "success_rate": float(successes.mean() * 100),
            "mean_episode_length": float(episode_lengths.mean()),
            "std_episode_length": float(episode_lengths.std()),
            "mean_total_reward": float(total_rewards.mean()),
            "std_total_reward": float(total_rewards.std()),
            "mean_touchdown_vy": mean_touchdown_vy,
            "std_touchdown_vy": std_touchdown_vy,
            "mean_n_fresh_observations": float(n_fresh.mean()),
            "std_n_fresh_observations": float(n_fresh.std()),
            "mean_n_fresh_observations_before_touchdown":float(n_fresh_before_touchdown.mean()),
            "std_n_fresh_observations_before_touchdown":float(n_fresh_before_touchdown.std()),
            "mean_n_fresh_observations_after_touchdown":float(n_fresh_after_touchdown.mean()),
            "std_n_fresh_observations_after_touchdown":float(n_fresh_after_touchdown.std()),
        })

    summary_rows.sort(key=lambda row: row["mean_fps"])

    summary_path = os.path.join(EVAL_ROOT, SUMMARY_CSV_NAME)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("_________________________________________________________")
    print(f"Run summary (from all */episodes.csv found under {EVAL_ROOT}):")
    for row in summary_rows:
        print(
            f"  {row['model']:<28} mean_fps={row['mean_fps']:>5.2f}  "
            f"n_episodes={row['n_episodes']:>3}  success_rate={row['success_rate']:6.2f}%  "
            f"reward mean={row['mean_total_reward']:.2f} std={row['std_total_reward']:.2f}  "
            f"mean fresh_obs={row['mean_n_fresh_observations']:.2f}  "
            f"mean_fresh_obs_b4_touchdown={row['mean_n_fresh_observations_before_touchdown']:.2f}  "
            f"mean_fresh_obs_after_touchdown={row['mean_n_fresh_observations_after_touchdown']:.2f}"
        )
    print(f"Summary written to: {summary_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", type=int, default=None, choices=FPS_CHOICES,
                        help="Fixed sensing rate in Hz. If given, bypasses the adaptive "
                            "sensing policy and forces this rate every step; if omitted, "
                            "the adaptive policy from --model is evaluated instead.")
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--nav-model", type=str, default=NAV_MODEL_PATH_DEFAULT,
                         help="Path to the frozen navigation/landing controller checkpoint.")
    parser.add_argument("--model", type=str, default=None,
                         help="Path to the trained adaptive sensing-policy checkpoint "
                              "(required unless --fixed is given).")
    parser.add_argument("--fc", type=float, required=True)
    parser.add_argument("--bud", type=float, required=True)
    
    #parser.add_argument("--seed", type=int, default=42)
    #parser.add_argument("--max-episode-steps", type=int, default=500)
    #parser.add_argument("--output", type=str, default=None, help="Policy-run folder name under eval/lunar_lander/. Auto-derived if omitted.")
    args = parser.parse_args()

    if args.fixed is None and args.model is None:
        parser.error("--model is required when --fixed is not provided")

    # Get arguments
    fixed_fps = args.fixed
    navigation_model = args.nav_model
    budget = args.bud
    frame_cost = args.fc

    # Calls summarize results at the end of the script
    atexit.register(summarize_results)

    # Creates LunarLander Env
    nav_model = NavModel(navigation_model, device=torch.device("cpu"))
    env = gym.make("LunarLander_VarFramerate", frame_cost=frame_cost, budget=budget)
    env.unwrapped.navigation_model = nav_model
    env = TimeLimit(env, max_episode_steps=500)

    if fixed_fps is not None:
        out_dir = f"{EVAL_ROOT}/fixed_{fixed_fps}Hz"
        model = None
        model_label = f"fixed_{fixed_fps}Hz"
    else:
        print("Loading FPS model...")
        checkpoint = torch.load(args.model, map_location='cpu')
        model = AgentEval(obs_dim=OBS_DIM, n_actions=ACTION_SPACE_LENGTH, lstm_hidden_size=LSTM_HIDDEN_SIZE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"FPS model loaded: {args.model}")
        model_label = f"adaptive_fc_{args.fc}"
        out_dir = f"{EVAL_ROOT}/adaptive_fc_{args.fc}_bud_{args.bud}"

    episode_rows = []
    for ep_index in range(args.n_episodes):
        episode_dir = f"{out_dir}/lap_{ep_index:02d}"
        seed = RUN_SEED + ep_index
        row = run_episode(env, fixed_fps, episode_dir, model, seed)
        row["episode_index"] = ep_index
        episode_rows.append(row)

        print(
            f"[{model_label}] Episode {ep_index}: "
            f"success={row['success']} "
            f"episode_length={row['episode_length']} "
            f"adaptive_reward={row['adaptive_reward']:.3f} "
            f"nav_reward={row['nav_reward']:.3f} "
            f"mean_fps={row['mean_fps']:.2f} "
            f"fresh_obs_ratio_b4_td={row['fresh_observation_ratio_b4_td']:.2f} "
            f"fresh_obs_ratio={row['fresh_observation_ratio']:.2f}"
        )

    env.close()

    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/episodes.csv", 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=EPISODE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(episode_rows)
    print(f"Episode results written to: {out_dir}/episodes.csv")

    config = {
        "mode": model_label,
        "model": args.model,
        "nav_model": args.nav_model,
        "frame_cost": frame_cost,
        "budget": budget,
        "fixed_fps": fixed_fps,
        "episodes": args.n_episodes,
        "initial seed": RUN_SEED,
        "max_episode_steps": 500,
        "device": "cpu",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(f"{out_dir}/config.json", "w") as file:
        json.dump(config, file, indent=2)

if __name__ == "__main__":
    main()
