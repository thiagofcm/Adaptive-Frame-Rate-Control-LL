"""
KF/debug_kf.py
──────────────
Standalone validation of the Kalman filter in envs/lunar_lander_var_fps_kf.py -- no PPO,
no adaptive-FPS policy involved. Runs one episode with a FIXED sensing FPS (chosen by this
script up front, not learned) using a frozen nav-model checkpoint as the low-level landing
controller, and produces plots of true state vs. KF estimate vs. a naive "held last sample"
baseline, KF uncertainty (trace P) over time, and the Kalman gain at each update tick.

Usage:
    python KF/debug_kf.py --config noiseless
    python KF/debug_kf.py --config max_staleness
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NTHREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)  # so `import envs....` resolves regardless of cwd

import argparse
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import envs.lunar_lander_var_fps_kf  # noqa: F401 -- registers "LunarLander_VarFramerate_KF"
from envs.lunar_lander_var_fps_kf import VIEWPORT_W, VIEWPORT_H, SCALE, LEG_DOWN, FPS

STATE_NAMES = ["x", "vx", "y", "vy", "angle", "angular_velocity"]

NAV_MODEL_PATH = os.path.join(
    REPO_ROOT, "runs/LunarLander_GaussianWind__train__windTrue_20.0_2.0_vert20.0_sns0.05__1__1785419028/model.pt"
)


# ══════════════════════════════════════════════════════
# Nav model (frozen, loaded from .pt) -- plain MLP, obs_dim=8, n_actions=4, matching
# experiments/navigation/train.py's Agent
# ══════════════════════════════════════════════════════

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class NavAgent(nn.Module):
    def __init__(self, obs_dim=8, n_actions=4):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )


class NavModel:
    def __init__(self, model_path, device):
        self.device = device
        checkpoint = torch.load(model_path, map_location=device)
        self.agent = NavAgent().to(device)
        sd = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        self.agent.load_state_dict(sd)
        self.agent.eval()

    def predict(self, obs, deterministic=True):
        obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = torch.argmax(self.agent.actor(obs_tensor), dim=-1)
        return action.cpu().numpy()[0], None


def get_true_state(u):
    """True 6-dim continuous state, pulled directly from the Box2D body -- NOT the
    returned observation (which, with the KF wired in, is the filtered estimate, not
    ground truth). Same scaling _physics_step uses, just computed independently here.

    Built in raw obs order [x, y, vx, vy, angle, angular_velocity] then permuted via
    u.kf_perm into the KF's own internal order [x, vx, y, vy, angle, angular_velocity]
    (matching STATE_NAMES/kf_x) -- kf_perm is a self-inverse transposition (swaps
    positions 1<->2), same one the env itself uses at its own obs<->KF boundaries.
    """
    pos = u.lander.position
    vel = u.lander.linearVelocity
    raw_order = np.array([
        (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
        (pos.y - (u.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
        vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
        vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
        u.lander.angle,
        20.0 * u.lander.angularVelocity / FPS,
    ], dtype=np.float64)
    return raw_order[u.kf_perm]


# ══════════════════════════════════════════════════════
# Configs
# ══════════════════════════════════════════════════════

CONFIGS = {
    # NOTE: Q is NOT actually gated by enable_wind -- envs/lunar_lander_var_fps_kf.py
    # builds kf_Q unconditionally from wind_power/vertical_wind_power/turbulence_power
    # in reset() (no enable_wind check near it). enable_wind only gates whether the
    # real disturbance FORCE gets applied to the lander in _physics_step(); it does not
    # zero out the filter's own assumed process-noise model. So this config still has a
    # nonzero Q (derived from wind_power=20/turbulence_power=2/vertical_wind_power=20
    # below) even though no real wind is injected -- P will still grow between updates
    # and K will be nonzero, same as any other config. Small measurement noise. The KF
    # should track truth closely if F/Q/H/R are wired correctly *and* the deterministic
    # constant-velocity+gravity+thrust model is a good fit -- note the dispersion jitter
    # in the real engine impulse isn't modeled in the KF's B@u term, so don't be
    # surprised if there's still a small residual drift: that's an accepted model-
    # mismatch simplification, not a wiring bug.
    "noiseless": dict(
        enable_wind=False, wind_power=20.0, turbulence_power=2.0, vertical_wind_power=20.0,
        sensor_noise_std=0.05, fixed_fps=50,
    ),
    # Wind on (matching the nav model's own training conditions), FPS pinned to 1 for
    # the whole episode -- longest possible stale stretches (50 physics ticks between
    # samples) to make KF-predict-through-staleness vs. naive-hold as visibly different
    # as possible.
    "max_staleness": dict(
        enable_wind=True, wind_power=20.0, turbulence_power=2.0, vertical_wind_power=20.0,
        sensor_noise_std=0.05, fixed_fps=1,
    ),
    # Wind on (same as max_staleness), but FPS pinned to 10 instead of 1 -- moderate
    # staleness (5 physics ticks between samples instead of 50), so both process noise
    # (real disturbance forces) and measurement noise (sensor_noise_std) are active,
    # but stale stretches are much shorter than max_staleness.
    "wind_fps10": dict(
        enable_wind=True, wind_power=20.0, turbulence_power=2.0, vertical_wind_power=20.0,
        sensor_noise_std=0.05, fixed_fps=10,
    ),
    # Same wind/noise settings, FPS pinned to 25 -- shorter stale stretches (2 physics
    # ticks between samples) than wind_fps10.
    "wind_fps25": dict(
        enable_wind=True, wind_power=20.0, turbulence_power=2.0, vertical_wind_power=20.0,
        sensor_noise_std=0.05, fixed_fps=25,
    ),
    # Same wind/noise settings, FPS pinned to 5 -- longer stale stretches (10 physics
    # ticks between samples) than wind_fps10.
    "wind_fps5": dict(
        enable_wind=True, wind_power=20.0, turbulence_power=2.0, vertical_wind_power=20.0,
        sensor_noise_std=0.05, fixed_fps=5,
    ),
}


# ══════════════════════════════════════════════════════
# Episode runner
# ══════════════════════════════════════════════════════

def add_gif_title(frame, seed, config_name, tick, sampled, grounded):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    title = f"seed={seed}  config={config_name}  tick={tick}"
    status = f"{'SAMPLE' if sampled else 'stale'}{'  GROUNDED' if grounded else ''}"
    draw.text((10, 10), title, fill=(255, 255, 255))
    draw.text((10, 26), status, fill=(0, 255, 0) if sampled else (255, 200, 0))
    return img


def run_episode(cfg, config_name, seed, max_steps, record_gif=False):
    render_mode = "rgb_array" if record_gif else None
    env = gym.make(
        "LunarLander_VarFramerate_KF",
        frame_cost=0.0, budget=1e9,
        enable_wind=cfg["enable_wind"], wind_power=cfg["wind_power"],
        turbulence_power=cfg["turbulence_power"], vertical_wind_power=cfg["vertical_wind_power"],
        sensor_noise_std=cfg["sensor_noise_std"],
        render_mode=render_mode,
    )
    u = env.unwrapped
    device = torch.device("cpu")
    u.navigation_model = NavModel(NAV_MODEL_PATH, device)

    fps_action = u.fps_choices.index(cfg["fixed_fps"])
    print(f"Fixed sensing FPS = {cfg['fixed_fps']} (action index {fps_action}, "
          f"obs_interval = {u.simulation_fps // cfg['fixed_fps']} physics ticks between samples)")

    obs, info = env.reset(seed=seed)

    log = {"true": [], "kf_x": [], "P_diag": [], "sampled": [], "grounded": [], "held": [], "K_diag": []}
    frames = [] if record_gif else None

    true0 = get_true_state(u)
    # Tick 0 has no sensor reading yet -- kf_x itself is seeded noiselessly at reset
    # (see envs/lunar_lander_var_fps_kf.py's reset()), so starting held from true state
    # here matches kf_x's own seeding, not an inconsistency.
    held_state = true0.copy()

    def record(sampled, tick):
        log["true"].append(get_true_state(u))
        log["kf_x"].append(u.kf_x.copy())
        log["P_diag"].append(np.diag(u.kf_P).copy())
        log["sampled"].append(sampled)
        grounded = bool(u.legs[0].ground_contact or u.legs[1].ground_contact)
        log["grounded"].append(grounded)
        log["held"].append(held_state.copy())
        log["K_diag"].append(np.diag(u.kf_K).copy() if u.kf_K is not None else np.full(6, np.nan))
        if record_gif:
            frames.append(add_gif_title(env.render(), seed, config_name, tick, sampled, grounded))

    record(sampled=True, tick=0)  # tick 0: the reset-seeded state (KF x == true state, P == 0)

    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < max_steps:
        obs, reward, terminated, truncated, info = env.step(fps_action)
        steps += 1

        sampled = bool(info["frame_consumed"])
        if sampled:
            # Naive-held baseline: what a system with NO Kalman filter but the SAME
            # noisy sensor would return -- the last noisy measurement z, not the
            # noiseless true state (see KF/eval_kf_quality.py for the same fix and why
            # using true state here was an unfair, artificially-clean baseline).
            held_state = u.kf_last_z.copy()
            print(f"  tick {steps:4d} | SAMPLE | K diag = {np.round(np.diag(u.kf_K), 4)}")

        record(sampled=sampled, tick=steps)

        if terminated or truncated:
            print(f"Episode ended at tick {steps} (terminated={terminated}, truncated={truncated})")

    env.close()
    for k in log:
        log[k] = np.array(log[k])
    return log, frames


# ══════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════

def _shade_ticks(ax, ticks, mask, color, alpha):
    for t in ticks[mask]:
        ax.axvspan(t - 0.5, t + 0.5, color=color, alpha=alpha, linewidth=0)


def plot_results(log, config_name, out_dir, seed):
    os.makedirs(out_dir, exist_ok=True)
    n = len(log["true"])
    ticks = np.arange(n)

    # --- true vs kf_x vs held, one subplot per dimension ---
    fig, axes = plt.subplots(6, 1, figsize=(13, 20), sharex=True)
    for i, name in enumerate(STATE_NAMES):
        ax = axes[i]
        _shade_ticks(ax, ticks, log["grounded"], "gray", 0.15)
        _shade_ticks(ax, ticks, log["sampled"], "green", 0.08)
        ax.plot(ticks, log["held"][:, i], label="held (naive)", color="tab:red", linewidth=1.0, linestyle=":")
        ax.plot(ticks, log["kf_x"][:, i], label="kf_x (predicted/fused)", color="tab:blue", linewidth=1.3, linestyle="--")
        ax.plot(ticks, log["true"][:, i], label="true", color="black", linewidth=1.3)
        ax.set_ylabel(name)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=3)
    axes[-1].set_xlabel("physics tick")
    fig.suptitle(f"True vs KF vs naive-held state -- config: {config_name}\n"
                 f"(green shading = sampling tick, gray shading = grounded)")
    fig.tight_layout()
    path1 = os.path.join(out_dir, f"state_overlay_seed{seed}_{config_name}.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)

    # --- trace(P) over time ---
    trace_P = log["P_diag"].sum(axis=1)
    fig2, ax = plt.subplots(figsize=(13, 4.5))
    _shade_ticks(ax, ticks, log["grounded"], "gray", 0.15)
    _shade_ticks(ax, ticks, log["sampled"], "green", 0.08)
    ax.plot(ticks, trace_P, color="tab:purple")
    ax.set_xlabel("physics tick")
    ax.set_ylabel("trace(P)")
    ax.set_title(f"KF uncertainty (trace P) -- config: {config_name}\n"
                 f"(green = sampling tick, gray = grounded; should sawtooth: grow while stale, drop at updates)")
    fig2.tight_layout()
    path2 = os.path.join(out_dir, f"trace_P_seed_{seed}_{config_name}.png")
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)

    # --- Kalman gain diag at update ticks ---
    fig3, ax = plt.subplots(figsize=(13, 4.5))
    sample_mask = log["sampled"]
    sample_ticks = ticks[sample_mask]
    for i, name in enumerate(STATE_NAMES):
        ax.plot(sample_ticks, log["K_diag"][sample_mask, i], label=name, marker="o", markersize=3)
    ax.set_xlabel("physics tick (sampling ticks only)")
    ax.set_ylabel("Kalman gain (diagonal)")
    ax.set_title(f"Kalman gain at each update -- config: {config_name}")
    ax.legend(fontsize=8, ncol=3)
    fig3.tight_layout()
    path3 = os.path.join(out_dir, f"kalman_gain_{config_name}.png")
    fig3.savefig(path3, dpi=150)
    plt.close(fig3)

    print(f"Plots saved: {path1}, {path2}, {path3}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(CONFIGS.keys()), required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default=os.path.join(REPO_ROOT, "KF", "debug_plots"))
    parser.add_argument("--gif", action="store_true",
                         help="also render the episode and save it as a GIF (seed + config in the filename, same output_dir as the plots)")
    parser.add_argument("--gif_fps", type=int, default=50,
                         help="GIF playback speed in frames/sec -- default 50 matches the physics tick rate (real-time)")
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    print(f"Config '{args.config}': {cfg}")
    log, frames = run_episode(cfg, args.config, seed=args.seed, max_steps=args.max_steps, record_gif=args.gif)
    plot_results(log, args.config, args.output_dir, args.seed)

    if args.gif:
        os.makedirs(args.output_dir, exist_ok=True)
        gif_path = os.path.join(args.output_dir, f"episode_seed{args.seed}_{args.config}.gif")
        duration_ms = int(1000 / args.gif_fps)
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
        print(f"GIF saved: {gif_path} ({len(frames)} frames)")
