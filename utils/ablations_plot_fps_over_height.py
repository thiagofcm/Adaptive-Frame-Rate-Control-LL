"""
combined_fps_height.py
──────────────────────
Combines FPS trace (top) and height profile (bottom)
into one figure with shared X axis.

Usage:
    python plot_fps_over_height.py \
        --model runs/.../model.pt \
        --nav_model runs/.../nav_model.pt \
        --lp 0.0 \
        --fc 0.0 \
        --n_ep 1
"""

import os
os.environ["OMP_NUM_THREADS"]     = "1"
os.environ["MKL_NUM_THREADS"]     = "1"
os.environ["OPENBLAS_NTHREADS"]   = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import torch.nn as nn
from torch.distributions.categorical import Categorical
import envs.lunar_lander_var_fps_simple_padd

# ══════════════════════════════════════════════════════
# Defines
# ══════════════════════════════════════════════════════
NAV_MODEL_PATH = "runs/LunarLander-v3__ppo__1__1779191150/model.pt"
FPS_CHOICES    = [1, 5, 10, 25, 50]
LSTM_HIDDEN_SIZE = 64
OBS_DIM          = 10
N_ACTIONS        = 5

# ── Style ──────────────────────────────────────────────
plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "figure.titlesize": 60,  # Figure title size.
    "font.size":        30,   # Default font size for all text elements unless overridden later.
    # "axes.titlesize":  60,   # Title size for axes titles (Ex: Timesteps, Reward).
    "axes.labelsize":  56,   # Controls axis label sizes: x-axis and y-axis labels (Ex: Timesteps, Reward).
    "xtick.labelsize": 50,   # Controls the size of tick numbers on the x-axis.
    "ytick.labelsize": 50,   # Controls the size of tick numbers on the y-axis.
    "legend.fontsize": 27,  
    "figure.dpi":      300, # Controls figure resolution (Dots Per Inch).
})

RUN_SEED = 42

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


# ══════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════

def evaluate(agent, nav_model, budget, frame_cost, fixed_fps,seed):

    env = gym.make("LunarLander_VarFramerate_SimplePadded",
                   frame_cost=frame_cost, budget=budget)
    env.unwrapped.navigation_model = nav_model
    env = TimeLimit(env, max_episode_steps=500)

    if not seed:
        random_num = random.randint(1, 100)
        seed = RUN_SEED + random_num

    obs, _ = env.reset(seed=seed)
    terminated  = False
    truncated   = False
    lstm_state  = make_initial_lstm_state()
    done        = False

    fps_per_timestep    = []
    height_per_timestep = []
    total_timesteps     = []
    step_count          = 0

    while not (terminated or truncated):
        if fixed_fps == 0.0:
            action, lstm_state = agent.predict(obs, lstm_state, done, deterministic=True)
        else:
            action = FPS_CHOICES.index(fixed_fps)

        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if done:
            lstm_state = make_initial_lstm_state()

        true_obs = env.unwrapped.current_obs
        fps_per_timestep.append(info["chosen_fps"])
        height_per_timestep.append(true_obs[1])
        total_timesteps.append(step_count)
        step_count += 1

    env.close()
    return total_timesteps, fps_per_timestep, height_per_timestep, seed


# ══════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════

def plot_combined(total_timesteps, fps_per_timestep, height_per_timestep,
                  seed, budget, fc, output_dir, fixed_fps):

    landing_timestep = None
    fig, (ax_fps, ax_height) = plt.subplots(
        2, 1, figsize=(15, 15), sharex=True,
        gridspec_kw={"hspace": 0.08}
    )

    ax_fps.plot(total_timesteps, fps_per_timestep, linewidth=3.5, label="Selected FPS", color="tab:blue")
    ax_fps.set_ylabel("FPS", labelpad=16)
    ax_fps.grid(True, alpha=0.9)
    for fps in FPS_CHOICES:
        ax_fps.axhline(y=fps, linestyle=":", linewidth=3.0, alpha=0.9, color="gray", label="Fixed FPS baselines" if fps == FPS_CHOICES[0] else None)
    ax_fps.set_yticks(FPS_CHOICES)
    ax_fps.set_yticklabels([str(fps) for fps in FPS_CHOICES], fontsize=40)
    ax_fps.set_ylim(0, 55)

    ax_height.plot(total_timesteps, height_per_timestep, linewidth=3.5, color="tab:orange") #label="Height")
    ax_height.set_xlabel("Timesteps", labelpad=10)
    ax_height.set_ylabel("Height", labelpad=30)
    ax_height.grid(True, alpha=0.9)
    ax_height.axhline(y=0, linestyle="--", linewidth=1.5, color="black",
                      alpha=0.8, label="Ground (height = 0)")

    # fig.suptitle(f"Adaptive Frame Rate Behavior")
    # fig.text(
    #     0.5, 0.90,
    #     f"Episode Seed={seed} | FC={fc} | Budget={budget}",
    #     ha="center",
    #     fontsize=24
    # )

    fig.suptitle(f"Ep Seed={seed} | Ablation D", y=0.94)

    for t, h in zip(total_timesteps, height_per_timestep):
        if h <= 0:
            landing_timestep = t
            break

    if landing_timestep is not None:
        ax_fps.axvline(x=landing_timestep, linestyle="--", linewidth=2,
                       color="red", label="Touchdown Time")
        ax_height.axvline(x=landing_timestep, linestyle="--", linewidth=2, color="red", label="Touchdown Time")

    #ax_fps.axvline.legend(loc="upper right")
    # Get legend entries from both subplots
    handles_fps, labels_fps = ax_fps.get_legend_handles_labels()
    handles_height, labels_height = ax_height.get_legend_handles_labels()

    # Combine them
    handles = handles_fps + handles_height
    labels = labels_fps + labels_height

    # Remove duplicated labels, e.g., Touchdown Instant
    unique = dict(zip(labels, handles))

    # One global legend outside the plots
    # fig.legend(
    #     unique.values(),
    #     unique.keys(),
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, 0.94),
    #     columnspacing=0.8,
    #     ncol=4,
    #     frameon=False,
    #     handlelength=1.0
    # )

    plt.tight_layout(rect=[0, 0, 1, 0.86])

    if fixed_fps == 0.0:
        filename = os.path.join(output_dir, f"combined_seed{seed}_BUD{budget}_FC{fc}.png")
    else:
        filename = os.path.join(output_dir, f"combined_seed{seed}_Fixed_FPS_{fixed_fps}.png")

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


# ══════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str,   required=True)
    parser.add_argument("--nav_model",  type=str,   default=NAV_MODEL_PATH)
    parser.add_argument("--budget",     type=float, required=True)
    parser.add_argument("--fc",         type=float, required=True)
    parser.add_argument("--n_ep",       type=int,   default=1)
    parser.add_argument("--fixed",      type=float, default=0.0)
    parser.add_argument("--seed",       type=int,   default=1, required=False)
    parser.add_argument("--output_dir", type=str,   default="result_plots/fps_over_height")
    args = parser.parse_args()

    device = torch.device("cpu")

    print("Loading nav model...")
    nav_model = NavModel(args.nav_model, device=device)
    print(f"Nav model loaded: {args.nav_model}")

    print("Loading agent...")
    checkpoint = torch.load(args.model, map_location=device)
    agent = AgentEval(obs_dim=OBS_DIM, n_actions=N_ACTIONS,
                      lstm_hidden_size=LSTM_HIDDEN_SIZE)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()
    print(f"Agent loaded: {args.model}")

    ep_results = {}
    for i in range(int(args.n_ep)):
        print(f"\nEvaluating episode {i+1}/{args.n_ep}")
        total_timesteps, fps_per_timestep, height_per_timestep, seed = evaluate(
            agent=agent,
            nav_model=nav_model,
            budget=args.budget,
            frame_cost=args.fc,
            fixed_fps=args.fixed,
            seed=args.seed,
        )
        ep_results[i] = total_timesteps, fps_per_timestep, height_per_timestep, seed

    print(f"\nPlotting {len(ep_results)} combined figures...")
    for idx in range(len(ep_results)):
        total_timesteps, fps_per_timestep, height_per_timestep, seed = ep_results[idx]
        plot_combined(
            total_timesteps=total_timesteps,
            fps_per_timestep=fps_per_timestep,
            height_per_timestep=height_per_timestep,
            seed=seed,
            budget=args.budget,
            fc=args.fc,
            output_dir=args.output_dir,
            fixed_fps=args.fixed,
        )

    print(f"\nDone! Plots saved to: {args.output_dir}/")