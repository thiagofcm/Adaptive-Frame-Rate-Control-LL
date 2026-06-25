import os
os.environ["OMP_NUM_THREADS"]    = "1"
os.environ["MKL_NUM_THREADS"]    = "1"
os.environ["OPENBLAS_NTHREADS"]  = "1"
os.environ["NUMEXPR_NUM_THREADS"]= "1"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
import sys
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
import argparse
import envs.lunar_lander_var_fps_simple_padd as lunar_lander_var_fps
import re
import torch.nn as nn
import csv
from multiprocessing import Pool, cpu_count
import functools
import traceback
from pathlib import Path


OBS_DIM = 10
ACTION_SPACE_LENGTH = 5
LSTM_HIDDEN_SIZE = 64
FPS_TO_ACTION = {1: 0, 5: 1, 10: 2, 25: 3, 50: 4}

# =========================
# USER INPUT
# =========================
N_EPISODES      = 100
N_RUNS          = 1
RUN_SEED        = 42
MAX_EVAL_WORKERS = 16
NAV_MODEL_PATH = "runs/LunarLander-v3__ppo__1__1779191150/model.pt"

# =========================
# SCORING WEIGHTS
# =========================
W_SUCCESS = 0.30
W_NAV     = 0.30
W_FRAMES  = 0.40


# ══════════════════════════════════════════════════════
# Model classes
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
        done   = done.reshape((-1, batch_size))
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
        obs_tensor  = torch.FloatTensor(obs).unsqueeze(0)
        done_tensor = torch.FloatTensor([float(done)])
        with torch.no_grad():
            hidden, lstm_state = self.get_states(obs_tensor, lstm_state, done_tensor)
            if deterministic:
                action = torch.argmax(self.actor(hidden), dim=-1)
            else:
                action = Categorical(logits=self.actor(hidden)).sample()
        return action.cpu().numpy()[0], lstm_state


    def make_initial_lstm_state():
        return (
            torch.zeros(1, 1, LSTM_HIDDEN_SIZE),
            torch.zeros(1, 1, LSTM_HIDDEN_SIZE),
        )

# =========================
# Utils
# =========================
def get_seeds(run, n_episodes):
    return [RUN_SEED + run * n_episodes + i for i in range(n_episodes)]

# =========================
# Evaluate one model for one run
# =========================
def evaluate_model_single_run(model, nav_model, frame_cost, budget, fixed):
    env = gym.make("LunarLander_VarFramerate_SimplePadded", frame_cost=frame_cost, budget=budget)
    env.unwrapped.navigation_model = nav_model
    env = TimeLimit(env, max_episode_steps=500)
    seeds = range(RUN_SEED, RUN_SEED + N_EPISODES) 

    episode_rewards      = []
    episode_nav_rewards  = []
    episode_frames       = []
    episode_vy           = []
    episode_success      = []
    episode_fps_traces   = []

    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        terminated, truncated = False, False


        # Reset LSTM state at the start of each episode
        lstm_state = (
            torch.zeros(1, 1, LSTM_HIDDEN_SIZE),
            torch.zeros(1, 1, LSTM_HIDDEN_SIZE),
        )
        done = False

        total_reward     = 0.0
        total_nav_reward = 0.0
        frame_count      = 0
        fps_trace        = []
        touchdown_vy     = None
        touchdown_flag   = False
        went_up_after    = False
        landed_in_flags  = False
        outside_flags_after_landing = False
        exceed_vy_vel = False
        prev_leg1 = False
        prev_leg2 = False

        while not (terminated or truncated):

            if fixed != 0:
                action = FPS_TO_ACTION[fixed]
            else:
                action, lstm_state = model.predict(obs, lstm_state, done, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            true_obs = env.unwrapped.current_obs
            
            total_reward     += reward
            total_nav_reward += info.get("nav_reward", 0.0)
            frame_count       = info["episode_frame_count"]
            fps_trace.append(info["chosen_fps"])

            leg1 = bool(true_obs[6])
            leg2 = bool(true_obs[7])
            leg_contact = leg1 or leg2

            # Detect first touchdown
            if leg_contact and not touchdown_flag:
                touchdown_vy    = abs(true_obs[3])
                touchdown_flag  = True
                landed_in_flags = (-0.2 < true_obs[0] < 0.2)
                exceed_vy_vel   = touchdown_vy > 0.8
                #print(f"  Ep {ep+1} | Touchdown → x={true_obs[0]:.3f} | vy={touchdown_vy:.4f} | in_flags={landed_in_flags}")

            # While grounded, check if drifts outside flags
            if touchdown_flag and leg_contact:
                if not (-0.2 < true_obs[0] < 0.2):
                    outside_flags_after_landing = True
                    #print(f"  Ep {ep+1} | Drifted outside flags → x={true_obs[0]:.3f}")

            both_grounded_prev = prev_leg1 and prev_leg2
            both_off_ground  = not leg1 and not leg2
            # Detect bounce
            if touchdown_flag and both_grounded_prev and both_off_ground:
                went_up_after = True
                #rint(f"  Ep {ep+1} | Bounce detected → y={true_obs[1]:.3f}")
            
            prev_leg1 = leg1
            prev_leg2 = leg2

        successful = landed_in_flags and not outside_flags_after_landing and not went_up_after and not exceed_vy_vel

        episode_rewards.append(total_reward)
        episode_nav_rewards.append(total_nav_reward)
        episode_frames.append(frame_count)
        episode_vy.append(touchdown_vy if touchdown_vy is not None else np.nan)
        episode_success.append(float(successful))
        episode_fps_traces.append(fps_trace)

        print(f"  Episode {seed - RUN_SEED + 1}/{N_EPISODES} | reward={total_reward:.2f} | success={successful} | frames={frame_count}")

    env.close()

    return {
        "rewards":      np.array(episode_rewards),
        "nav_rewards":  np.array(episode_nav_rewards),
        "frames":       np.array(episode_frames),
        "vy":           np.array(episode_vy),
        "success":      np.array(episode_success),
        "fps_traces":   episode_fps_traces,
    }

# =========================
# Main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--fc", type=float, required=True)
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--fixed", type=float, default=0.0, required=False)
    args = parser.parse_args()

    model_path     = Path(args.model)
    frame_cost     = args.fc
    budget         = args.budget  # Budget of frames before landing
    parent_folder  = model_path.parent
    results_folder = "score_results"
    output_dir     = os.path.join(results_folder, parent_folder)
    fixed          = args.fixed
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cpu")

    # Load Nav Model
    print("Loading nav model...")
    nav_model = NavModel(NAV_MODEL_PATH, device=device)
    print(f"Nav model loaded: {NAV_MODEL_PATH}")

    # Load FPS Model
    print("Loading FPS model...")
    model = torch.load(args.model, map_location=device)
    agent = AgentEval(obs_dim=OBS_DIM, n_actions=ACTION_SPACE_LENGTH,lstm_hidden_size=LSTM_HIDDEN_SIZE)
    agent.load_state_dict(model["model_state_dict"])
    agent.eval()
    print(f"FPS model loaded: {model_path}")

    print(f"\nStarting evaluation: {N_RUNS} runs x {N_EPISODES} episodes each")
    print(f"Seeds: {RUN_SEED} → {RUN_SEED + N_EPISODES - 1}")

    # ── Run N_RUNS times, each with a non-overlapping seed block ──────────────
    run_metrics = {
        "rewards":     [],
        "nav_rewards": [],
        "frames":      [],
        "vy":          [],
        "success":     [],
    }

    for run in range(N_RUNS):
        print(f"\nRun {run+1}/{N_RUNS}")
        results = evaluate_model_single_run(agent, nav_model, frame_cost, budget, fixed)  # a matrix containing the result metrics of 100 different episodes
        run_metrics["rewards"].append(results["rewards"])                          # append the results into a matrix containing all the 10 runs (rows: RUNS (10), columns: EPISODES (100))
        run_metrics["nav_rewards"].append(results["nav_rewards"])
        run_metrics["frames"].append(results["frames"])
        run_metrics["vy"].append(results["vy"])
        run_metrics["success"].append(results["success"])

    for key in run_metrics:
        run_metrics[key] = np.array(run_metrics[key])

    # mean per episode across runs → (N_EPISODES,), then average those
    rew_m,  rew_std  = run_metrics["rewards"].mean(axis=0).mean(),              run_metrics["rewards"].std(axis=0).mean()
    nav_m,  nav_std  = run_metrics["nav_rewards"].mean(axis=0).mean(),          run_metrics["nav_rewards"].std(axis=0).mean()
    frm_m,  frm_std  = run_metrics["frames"].mean(axis=0).mean(),               run_metrics["frames"].std(axis=0).mean()
    vy_m,   vy_std   = np.nanmean(run_metrics["vy"], axis=0).mean(),            np.nanstd(run_metrics["vy"], axis=0).mean()
    succ_m, succ_std = run_metrics["success"].mean(axis=0).mean() * 100,        run_metrics["success"].std(axis=0).mean() * 100
                
    # ── Save CSV ───────────────────────────────────────────────────────────────
    if fixed > 0:
        output_dir = "score_results/runs"
        csv_path = os.path.join(output_dir, f"eval_results_fixed_{fixed}.csv")
    else:
        csv_path = os.path.join(output_dir, "eval_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow([
            "model", "frame_cost", "budget", "n_runs", "n_episodes",
            "reward_mean",     "reward_std",
            "nav_reward_mean", "nav_reward_std",
            "frames_mean",     "frames_std",
            "vy_mean",         "vy_std",
            "success_pct_mean","success_pct_std",
        ])
        # One summary row
        writer.writerow([
            str(model_path), frame_cost, budget, N_RUNS, N_EPISODES,
            f"{rew_m:.4f}",  f"{rew_std:.4f}",
            f"{nav_m:.4f}",  f"{nav_std:.4f}",
            f"{frm_m:.2f}",  f"{frm_std:.2f}",
            f"{vy_m:.4f}",   f"{vy_std:.4f}",
            f"{succ_m:.2f}", f"{succ_std:.2f}",
        ])
        # Per-run detail rows
        # writer.writerow([])
        # writer.writerow(["--- per-run breakdown ---"])
        # writer.writerow(["run", "reward", "nav_reward", "frames", "vy", "success_pct"])
        # for i in range(N_RUNS):
        #     writer.writerow([
        #         i + 1,
        #         f"{run_metrics['rewards'][i]:.4f}",
        #         f"{run_metrics['nav_rewards'][i]:.4f}",
        #         f"{run_metrics['frames'][i]:.2f}",
        #         f"{run_metrics['vy'][i]:.4f}",
        #         f"{run_metrics['success'][i]:.2f}",
        #     ])

    print(f"\nResults saved → {csv_path}")

    # ── Console summary ────────────────────────────────────────────────────────
    print("\n===== Evaluation Summary =====")
    print(f"Model        : {model_path}")
    print(f"Runs x Eps   : {N_RUNS} x {N_EPISODES} = {N_RUNS*N_EPISODES} total episodes")
    print(f"Reward       : {rew_m:.4f} ± {rew_std:.4f}")
    print(f"Nav Reward   : {nav_m:.4f} ± {nav_std:.4f}")
    print(f"Frames       : {frm_m:.2f} ± {frm_std:.2f}")
    print(f"Touchdown vy : {vy_m:.4f} ± {vy_std:.4f}")
    print(f"Success      : {succ_m:.2f}% ± {succ_std:.2f}%")