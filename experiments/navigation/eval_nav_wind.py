# running: python -m experiments.navigation.eval_nav_wind --model runs/LunarLander_GaussianWind__train__windFalse_15.0_1.5__1__1785187605/model.pt --enable-wind --wind-power 80.0 --turbulence-power 0.0 --vertical-wind-power 0.0 --vy-thr

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
import torch.nn as nn
import csv
from pathlib import Path
import envs.lunar_lander_gaussian_wind  # noqa: F401 -- import needed to register "LunarLander_GaussianWind"
from envs.lunar_lander_gaussian_wind import VIEWPORT_W, VIEWPORT_H, SCALE, LEG_DOWN, FPS


OBS_DIM = 8
ACTION_SPACE_LENGTH = 4

# =========================
# USER INPUT
# =========================
N_EPISODES = 1000
N_RUNS     = 1
RUN_SEED   = 42


# ══════════════════════════════════════════════════════
# Model class
# ══════════════════════════════════════════════════════

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class AgentEval(nn.Module):
    """Matches exactly the Agent class in train.py -- plain MLP, no LSTM (the nav model
    acts on the raw obs every physics tick, there's no adaptive-FPS staleness here)."""
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

    def predict(self, obs, deterministic=True):
        obs_tensor = torch.Tensor(obs).unsqueeze(0)
        with torch.no_grad():
            action = torch.argmax(self.actor(obs_tensor), dim=-1)
        return action.cpu().numpy()[0]


# =========================
# Evaluate one model for one run
# =========================
def get_true_state8(u):
    """True 8-dim state read directly from Box2D -- NOT the returned observation,
    which carries injected sensor_noise_std measurement noise once that's > 0. Success/
    touchdown scoring must use this, not the noisy obs, or near-threshold episodes
    (landed_in_flags's +-0.2 bound, the vy_thr crossing) get randomly flipped by
    measurement noise rather than reflecting the actual landing -- same principle as
    experiments/lunar_lander_var_fps_kf/eval.py's true_obs = env.unwrapped.current_obs."""
    pos = u.lander.position
    vel = u.lander.linearVelocity
    return np.array([
        (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
        (pos.y - (u.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
        vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
        vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
        u.lander.angle,
        20.0 * u.lander.angularVelocity / FPS,
        1.0 if u.legs[0].ground_contact else 0.0,
        1.0 if u.legs[1].ground_contact else 0.0,
    ])


def evaluate_model_single_run(model, enable_wind, wind_power, turbulence_power, vertical_wind_power, sensor_noise_std, vy_thr, max_episode_steps):
    env = gym.make("LunarLander_GaussianWind", enable_wind=enable_wind,
                    wind_power=wind_power, turbulence_power=turbulence_power,
                    vertical_wind_power=vertical_wind_power, sensor_noise_std=sensor_noise_std)
    # LunarLander_GaussianWind has no max_episode_steps at registration -- without this,
    # an episode that never crashes or lands under wind (quite possible for a nav model
    # that was never trained to handle wind) runs forever.
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    u = env.unwrapped
    seeds = range(RUN_SEED, RUN_SEED + N_EPISODES)

    episode_rewards = []
    episode_steps   = []
    episode_vy      = []
    episode_success = []

    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        terminated, truncated = False, False

        total_reward   = 0.0
        step_count     = 0
        touchdown_vy   = None
        touchdown_flag = False
        went_up_after   = False
        landed_in_flags = False
        outside_flags_after_landing = False
        exceed_vy_vel = False
        prev_leg1 = False
        prev_leg2 = False

        while not (terminated or truncated):
            action = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            step_count   += 1

            true_obs = get_true_state8(u)
            leg1 = bool(true_obs[6])
            leg2 = bool(true_obs[7])
            leg_contact = leg1 or leg2

            # Detect first touchdown
            if leg_contact and not touchdown_flag:
                touchdown_vy    = abs(true_obs[3])
                touchdown_flag  = True
                landed_in_flags = (-0.2 < true_obs[0] < 0.2)
                exceed_vy_vel   = touchdown_vy > vy_thr

            # While grounded, check if drifts outside flags
            if touchdown_flag and leg_contact:
                if not (-0.2 < true_obs[0] < 0.2):
                    outside_flags_after_landing = True

            both_grounded_prev = prev_leg1 and prev_leg2
            both_off_ground = not leg1 and not leg2
            # Detect bounce
            if touchdown_flag and both_grounded_prev and both_off_ground:
                went_up_after = True

            prev_leg1 = leg1
            prev_leg2 = leg2

        successful = landed_in_flags and not outside_flags_after_landing and not went_up_after and not exceed_vy_vel

        episode_rewards.append(total_reward)
        episode_steps.append(step_count)
        episode_vy.append(touchdown_vy if touchdown_vy is not None and successful else np.nan)
        episode_success.append(float(successful))

        print(f"  Episode {seed - RUN_SEED + 1}/{N_EPISODES} | reward={total_reward:.2f} | success={successful} | steps={step_count}")

    env.close()

    return {
        "rewards": np.array(episode_rewards),
        "steps":   np.array(episode_steps),
        "vy":      np.array(episode_vy),
        "success": np.array(episode_success),
    }

# =========================
# Main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--enable-wind", dest="enable_wind", action="store_true", default=True,
                         help="toggle the per-tick Gaussian wind/turbulence disturbance (on by default)")
    parser.add_argument("--no-enable-wind", dest="enable_wind", action="store_false")
    parser.add_argument("--wind-power",       type=float, default=20.0, help="std-dev of the per-tick Gaussian force")
    parser.add_argument("--turbulence-power", type=float, default=2.0,  help="std-dev of the per-tick Gaussian torque")
    parser.add_argument("--vertical-wind-power", type=float, default=20.0, help="std-dev of the per-tick Gaussian vertical (y/vy) force")
    parser.add_argument("--sensor-noise-std", type=float, default=0.05, help="std-dev of Gaussian measurement noise added to the 6 continuous obs dims (0.0 = off)")
    parser.add_argument("--vy_thr", type=float, default=0.5, required=False, help="touchdown vy threshold for success (m/s)")
    parser.add_argument("--max_episode_steps", type=int, default=500, required=False)
    args = parser.parse_args()

    model_path     = Path(args.model)
    parent_folder  = model_path.parent
    results_folder = "score_results"
    output_dir     = os.path.join(results_folder, parent_folder)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cpu")

    # Load nav model
    print("Loading nav model...")
    checkpoint = torch.load(args.model, map_location=device)
    agent = AgentEval(obs_dim=OBS_DIM, n_actions=ACTION_SPACE_LENGTH)
    sd = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    agent.load_state_dict(sd)
    agent.eval()
    print(f"Nav model loaded: {model_path}")

    print(f"\nStarting evaluation: {N_RUNS} runs x {N_EPISODES} episodes each")
    print(f"Seeds: {RUN_SEED} → {RUN_SEED + N_EPISODES - 1}")
    print(f"Wind: enable={args.enable_wind} wind_power={args.wind_power} turbulence_power={args.turbulence_power} vertical_wind_power={args.vertical_wind_power} sensor_noise_std={args.sensor_noise_std}")

    # ── Run N_RUNS times, each with a non-overlapping seed block ──────────────
    run_metrics = {
        "rewards": [],
        "steps":   [],
        "vy":      [],
        "success": [],
    }

    for run in range(N_RUNS):
        print(f"\nRun {run+1}/{N_RUNS}")
        results = evaluate_model_single_run(agent, args.enable_wind, args.wind_power, args.turbulence_power, args.vertical_wind_power, args.sensor_noise_std, args.vy_thr, args.max_episode_steps)
        run_metrics["rewards"].append(results["rewards"])
        run_metrics["steps"].append(results["steps"])
        run_metrics["vy"].append(results["vy"])
        run_metrics["success"].append(results["success"])

    for key in run_metrics:
        run_metrics[key] = np.array(run_metrics[key])

    # Mean/std over the FULL (n_runs, n_episodes) array at once -- NOT a
    # mean(axis=0)/std(axis=0) followed by a second .mean(): with N_RUNS=1 (always, in
    # this codebase), that axis=0 reduction is a single-sample reduction per episode --
    # the mean is a harmless no-op, but the std is *always exactly 0.0* by definition
    # (std of one number), regardless of the actual episode-to-episode spread. That
    # made every std reported here meaningless (structurally always 0), not just
    # "small" -- this reduces once over every episode across all runs, giving the real
    # episode-to-episode variability instead.
    rew_m,  rew_std  = run_metrics["rewards"].mean(),   run_metrics["rewards"].std()
    stp_m,  stp_std  = run_metrics["steps"].mean(),     run_metrics["steps"].std()
    # np.nanmean/nanstd: touchdown_vy is nan for episodes that never touch down (crashed
    # out of bounds / timed out airborne) -- must be excluded, not averaged in as 0.
    vy_m,   vy_std   = np.nanmean(run_metrics["vy"]), np.nanstd(run_metrics["vy"])
    succ_m = run_metrics["success"].mean() * 100

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(
        output_dir,
        f"eval_results_wind_hor{args.wind_power}_vert_{args.vertical_wind_power}_turb_{args.turbulence_power}_sns{args.sensor_noise_std}_vy_thr_{args.vy_thr}.csv",
    )
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "enable_wind", "wind_power", "turbulence_power", "vertical_wind_power", "sensor_noise_std", "n_runs", "n_episodes",
            "reward_mean",  "reward_std",
            "steps_mean",   "steps_std",
            "vy_mean",      "vy_std",
            "success_pct_mean",
        ])
        writer.writerow([
            str(model_path), args.enable_wind, args.wind_power, args.turbulence_power, args.vertical_wind_power, args.sensor_noise_std, N_RUNS, N_EPISODES,
            f"{rew_m:.4f}",  f"{rew_std:.4f}",
            f"{stp_m:.2f}",  f"{stp_std:.2f}",
            f"{vy_m:.4f}",   f"{vy_std:.4f}",
            f"{succ_m:.2f}",
        ])

    print(f"\nResults saved → {csv_path}")

    # ── Console summary ────────────────────────────────────────────────────────
    print("\n===== Evaluation Summary =====")
    print(f"Model        : {model_path}")
    print(f"Wind         : enable={args.enable_wind} wind_power={args.wind_power} turbulence_power={args.turbulence_power} vertical_wind_power={args.vertical_wind_power} sensor_noise_std={args.sensor_noise_std}")
    print(f"Runs x Eps   : {N_RUNS} x {N_EPISODES} = {N_RUNS*N_EPISODES} total episodes")
    print(f"Reward       : {rew_m:.4f} ± {rew_std:.4f}")
    print(f"Steps        : {stp_m:.2f} ± {stp_std:.2f}")
    print(f"Touchdown vy : {vy_m:.4f} ± {vy_std:.4f}")
    print(f"Success      : {succ_m:.2f}%")
