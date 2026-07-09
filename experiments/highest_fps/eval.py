import os
os.environ["OMP_NUM_THREADS"]    = "1"
os.environ["MKL_NUM_THREADS"]    = "1"
os.environ["OPENBLAS_NTHREADS"]  = "1"
os.environ["NUMEXPR_NUM_THREADS"]= "1"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import argparse
from experiments.highest_fps.classes import Agent, NavModel
import envs.lunar_lander_highest_fps as LunarLander_HighestFPS
import torch.nn as nn
import csv
from pathlib import Path

ACTION_SPACE_LENGTH = 4
FPS_TO_ACTION = {1: 0, 5: 1, 10: 2, 50: 3}

N_EPISODES      = 2
N_RUNS          = 1
RUN_SEED        = 42
NAV_MODEL_PATH  = "experiments/navigation/runs/LunarLander-v3__ppo__1__1779191150/model.pt"

W_SUCCESS = 0.30
W_NAV     = 0.30
W_FRAMES  = 0.40

# Utils
def get_seeds(run, n_episodes):
    return [RUN_SEED + run * n_episodes + i for i in range(n_episodes)]

def make_eval_env():
    def thunk():
        env = gym.make("LunarLander_HighestFPS")
        env.unwrapped.navigation_model = nav_model
        env = TimeLimit(env, max_episode_steps=500)
        return env
    return thunk

# Evaluate one model for one run
def evaluate_model_single_run(model, nav_model, fixed):
    env = gym.make("LunarLander_HighestFPS")
    env.unwrapped.navigation_model = nav_model
    env = TimeLimit(env, max_episode_steps=500)
    seeds = range(RUN_SEED, RUN_SEED + N_EPISODES) 

    episode_rewards      = []
    episode_nav_rewards  = []
    episode_frames       = []
    episode_vy           = []
    episode_success      = []
    total_mean_chosen_fps= []

    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        terminated, truncated = False, False
        done = False

        total_reward     = 0.0
        total_nav_reward = 0.0
        frame_count      = 0
        choosen_fps = []
        episode_mean_chosen_fps   = []
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
                action, _ = model.predict(obs, deterministic=True)
                chosen_fps = env.unwrapped.fps_choices[action]
                print(f"  Ep {seed - RUN_SEED + 1}/{N_EPISODES} | Chosen FPS: {chosen_fps}")

            obs, reward, terminated, truncated, info = env.step(action)
            true_obs = env.unwrapped.current_obs
            
            total_reward     += reward
            total_nav_reward += info.get("nav_reward", 0.0)
            frame_count       = info["episode_frame_count"]
            choosen_fps.append(info["chosen_fps"])

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
        episode_mean_chosen_fps.append(np.mean(choosen_fps) if choosen_fps else np.nan)

        print(f"  Episode {seed - RUN_SEED + 1}/{N_EPISODES} | reward={total_reward:.2f} | success={successful} | frames={frame_count} | Mean Chosen FPS: {episode_mean_chosen_fps[-1]:.2f}")

    env.close()

    return {
        "rewards":      np.array(episode_rewards),
        "nav_rewards":  np.array(episode_nav_rewards),
        "frames":       np.array(episode_frames),
        "vy":           np.array(episode_vy),
        "success":      np.array(episode_success),
        "mean_chosen_fps":   np.array(episode_mean_chosen_fps),
    }

if __name__ == "__main__":
    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--fixed", type=float, default=0.0, required=False)
    args = parser.parse_args()

    # Setup output directory
    model_path     = Path(args.model)
    parent_folder  = model_path.parent
    results_folder = "eval_results"
    output_dir     = os.path.join(parent_folder,results_folder)
    fixed          = args.fixed
    os.makedirs(output_dir, exist_ok=True)

    # Setup device
    device = torch.device("cpu")

    # Load Nav Model
    print("Loading nav model...")
    nav_model = NavModel(NAV_MODEL_PATH, device=device)
    print(f"Nav model loaded: {NAV_MODEL_PATH}")

    # Load FPS Model
    print("Loading FPS model...")
    model = torch.load(args.model, map_location=device)
    envs = gym.vector.SyncVectorEnv([make_eval_env()])
    agent = Agent(envs)
    agent.load_state_dict(model["model_state_dict"])
    agent.eval()
    print(f"FPS model loaded: {model_path}")
    print(f"\nStarting evaluation: {N_RUNS} runs x {N_EPISODES} episodes each")
    print(f"Seeds: {RUN_SEED} → {RUN_SEED + N_EPISODES - 1}")

    run_metrics = {
        "rewards":          [],
        "nav_rewards":      [],
        "frames":           [],
        "vy":               [],
        "success":          [],
        "mean_chosen_fps":  [],
    }

    for run in range(N_RUNS):
        print(f"\nRun {run+1}/{N_RUNS}")
        results = evaluate_model_single_run(agent, nav_model, fixed)  # a matrix containing the result metrics of 100 different episodes
        run_metrics["rewards"].append(results["rewards"])                          # append the results into a matrix containing all the 10 runs (rows: RUNS (10), columns: EPISODES (100))
        run_metrics["nav_rewards"].append(results["nav_rewards"])
        run_metrics["frames"].append(results["frames"])
        run_metrics["vy"].append(results["vy"])
        run_metrics["success"].append(results["success"])
        run_metrics["mean_chosen_fps"].append(results["mean_chosen_fps"])

    for key in run_metrics:
        run_metrics[key] = np.array(run_metrics[key])

    # mean per episode across runs → (N_EPISODES,), then average those
    rew_m,  rew_std  = run_metrics["rewards"].mean(axis=0).mean(),              run_metrics["rewards"].std(axis=0).mean()
    nav_m,  nav_std  = run_metrics["nav_rewards"].mean(axis=0).mean(),          run_metrics["nav_rewards"].std(axis=0).mean()
    frm_m,  frm_std  = run_metrics["frames"].mean(axis=0).mean(),               run_metrics["frames"].std(axis=0).mean()
    vy_m,   vy_std   = np.nanmean(run_metrics["vy"], axis=0).mean(),            np.nanstd(run_metrics["vy"], axis=0).mean()
    succ_m, succ_std = run_metrics["success"].mean(axis=0).mean() * 100,        run_metrics["success"].std(axis=0).mean() * 100
    fps_m,  fps_std  = run_metrics["mean_chosen_fps"].mean(axis=0).mean(),     run_metrics["mean_chosen_fps"].std(axis=0).mean()

    # ── Save CSV ───────────────────────────────────────────────────────────────
    if fixed > 0:
        csv_path = os.path.join(output_dir, f"eval_results_fixed_{fixed}.csv")
    else:
        csv_path = os.path.join(output_dir, "eval_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow([
            "model", "n_runs", "n_episodes",
            "reward_mean",     "reward_std",
            "nav_reward_mean", "nav_reward_std",
            "frames_mean",     "frames_std",
            "vy_mean",         "vy_std",
            "success_pct_mean","success_pct_std",
            "mean_chosen_fps_mean", "mean_chosen_fps_std",
        ])
        # One summary row
        writer.writerow([
            str(model_path), N_RUNS, N_EPISODES,
            f"{rew_m:.4f}",  f"{rew_std:.4f}",
            f"{nav_m:.4f}",  f"{nav_std:.4f}",
            f"{frm_m:.2f}",  f"{frm_std:.2f}",
            f"{vy_m:.4f}",   f"{vy_std:.4f}",
            f"{succ_m:.2f}", f"{succ_std:.2f}",
            f"{fps_m:.2f}",  f"{fps_std:.2f}",
        ])

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
    print(f"Mean Chosen FPS : {fps_m:.2f} ± {fps_std:.2f}")