"""
demo_nav_wind.py
────────────────
Loads a trained CleanRL PPO agent (.pt file) and runs one episode
on LunarLander_GaussianWind (same 8-dim obs / Discrete(4) as LunarLander-v3, but with
i.i.d. Gaussian force/torque process noise instead of no wind), saving a GIF of the
performance.

Usage:
    python demo_nav_wind.py --model_path runs/.../model.pt --output_path landing_wind.gif
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import envs.lunar_lander_gaussian_wind  # noqa: F401 -- import needed to register "LunarLander_GaussianWind"
from PIL import Image, ImageDraw


# ──────────────────────────────────────────────
# Must match exactly the Agent class in train.py
# ──────────────────────────────────────────────
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )

    def get_action(self, x):
        logits = self.actor(x)
        # deterministic — pick the action with highest logit
        return torch.argmax(logits, dim=-1)


def add_title(frame, seed, wind_power, turbulence_power, vertical_wind_power, sensor_noise_std):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    title = f"seed={seed}  wind_power={wind_power}  turbulence_power={turbulence_power}  vertical_wind_power={vertical_wind_power}  sensor_noise_std={sensor_noise_std}"
    draw.text((10, 10), title, fill=(255, 255, 255))
    return img


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  type=str, required=True,  help="Path to .pt model file")
    parser.add_argument("--output_path", type=str, default="landing_wind.gif", help="Output GIF path")
    parser.add_argument("--fps",         type=int, default=50,     help="GIF frames per second")
    parser.add_argument("--max_steps",   type=int, default=500,   help="Max steps per episode")
    parser.add_argument("--enable-wind", dest="enable_wind", action="store_true", default=True,
                         help="toggle the per-tick Gaussian wind/turbulence disturbance (on by default -- this script exists to demo it)")
    parser.add_argument("--no-enable-wind", dest="enable_wind", action="store_false")
    parser.add_argument("--wind-power",        type=float, default=40.0, help="std-dev of the per-tick Gaussian force")
    parser.add_argument("--turbulence-power",  type=float, default=3.0,  help="std-dev of the per-tick Gaussian torque")
    parser.add_argument("--vertical-wind-power", type=float, default=30.0, help="std-dev of the per-tick Gaussian vertical (y/vy) force")
    parser.add_argument("--sensor-noise-std", type=float, default=0.0, help="std-dev of Gaussian measurement noise added to the 6 continuous obs dims (0.0 = off)")
    parser.add_argument("--seed",        type=int, default=None,   help="env reset seed (omit for a random episode)")
    args = parser.parse_args()

    device = torch.device("cpu")  # cpu is fine for evaluation

    # ── Environment ────────────────────────────
    env = gym.make("LunarLander_GaussianWind", render_mode="rgb_array",
                    enable_wind=args.enable_wind, wind_power=args.wind_power,
                    turbulence_power=args.turbulence_power,
                    vertical_wind_power=args.vertical_wind_power,
                    sensor_noise_std=args.sensor_noise_std)
    obs_dim   = env.observation_space.shape[0]   # 8
    n_actions = env.action_space.n               # 4

    # ── Load model ─────────────────────────────
    agent = Agent(obs_dim, n_actions).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)

    # handle both plain state_dict and checkpoint dict
    if "model_state_dict" in checkpoint:
        agent.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from step {checkpoint.get('global_step', '?')}")
    else:
        agent.load_state_dict(checkpoint)
        print("Loaded model weights")

    agent.eval()

    # ── Run one episode ────────────────────────
    # Resolve the seed up front (rather than letting reset(seed=None) pick one
    # internally) so it's always a concrete value we can burn into the GIF title.
    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)

    # Bake seed/wind/turbulence into the filename (preserving whatever directory/name/
    # extension --output_path already specifies) so GIFs from different runs don't
    # collide and are identifiable without opening them.
    base, ext = os.path.splitext(args.output_path)
    output_path = f"{base}_seed{seed}_wind{args.wind_power}_turb{args.turbulence_power}_vert{args.vertical_wind_power}_sns{args.sensor_noise_std}{ext}"

    obs, _ = env.reset(seed=seed)
    frames = []
    total_reward = 0.0
    terminated = False
    truncated  = False
    step = 0

    print("Running episode...")

    while not (terminated or truncated) and step < args.max_steps:
        # render frame
        frame = env.render()
        frames.append(add_title(frame, seed, args.wind_power, args.turbulence_power, args.vertical_wind_power, args.sensor_noise_std))

        # get action — deterministic
        obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action = agent.get_action(obs_tensor)
        action = action.item()

        # step env
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

    # capture final frame
    frame = env.render()
    frames.append(add_title(frame, seed, args.wind_power, args.turbulence_power, args.vertical_wind_power, args.sensor_noise_std))
    env.close()
    print(f"Seed used: {seed}")

    print(f"Episode finished in {step} steps | Total reward: {total_reward:.2f}")

    # ── Save GIF ───────────────────────────────
    duration_ms = int(1000 / args.fps)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,  # loop forever
    )
    print(f"GIF saved to: {output_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
