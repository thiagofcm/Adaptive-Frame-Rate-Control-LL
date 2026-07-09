"""
evaluate_lunar.py
─────────────────
Loads a trained CleanRL PPO agent (.pt file) and runs one episode
on LunarLander-v3, saving a GIF of the performance.

Usage:
    python evaluate_lunar.py --model_path runs/.../model.pt --output_path landing.gif
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from PIL import Image


# ──────────────────────────────────────────────
# Must match exactly the Agent class in ppo.py
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


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  type=str, required=True,  help="Path to .pt model file")
    parser.add_argument("--output_path", type=str, default="landing.gif", help="Output GIF path")
    parser.add_argument("--fps",         type=int, default=50,     help="GIF frames per second")
    parser.add_argument("--max_steps",   type=int, default=1000,   help="Max steps per episode")
    args = parser.parse_args()

    device = torch.device("cpu")  # cpu is fine for evaluation

    # ── Environment ────────────────────────────
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
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
    obs, _ = env.reset()
    frames = []
    total_reward = 0.0
    terminated = False
    truncated  = False
    step = 0

    print("Running episode...")

    while not (terminated or truncated) and step < args.max_steps:
        # render frame
        frame = env.render()
        frames.append(Image.fromarray(frame))

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
    frames.append(Image.fromarray(frame))
    env.close()

    print(f"Episode finished in {step} steps | Total reward: {total_reward:.2f}")

    # ── Save GIF ───────────────────────────────
    duration_ms = int(1000 / args.fps)
    frames[0].save(
        args.output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,  # loop forever
    )
    print(f"GIF saved to: {args.output_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
