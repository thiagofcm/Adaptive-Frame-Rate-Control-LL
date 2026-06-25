import gymnasium as gym
import matplotlib.pyplot as plt
import envs.lunar_lander_var_fps_simple_padd as lunar_lander_var_fps
import numpy as np
import cv2
import imageio
import argparse
from gymnasium.wrappers import TimeLimit
import os
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

NAV_MODEL_PATH = "runs/LunarLander-v3__ppo__1__1779191150/model.pt"
GIF_FPS = 30
FPS_CHOICES = [1, 5, 10, 25, 50]
LSTM_HIDDEN_SIZE = 64
OBS_DIM = 10
N_ACTIONS = 5

# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def smooth(data, window=10):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def add_fps_overlay(frame, chosen_fps, step):
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # cv2.putText(frame_bgr, f"FPS: {chosen_fps}", (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    # cv2.putText(frame_bgr, f"Step: {step}", (10, 60),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def save_gif(frames, path, gif_fps=30):
    imageio.mimsave(path, frames, fps=gif_fps)
    print(f"GIF saved to: {path}")


def make_initial_lstm_state():
    return (
        torch.zeros(1, 1, LSTM_HIDDEN_SIZE),
        torch.zeros(1, 1, LSTM_HIDDEN_SIZE),
    )

# ------------------------------------------------------------------ #
# Nav model (frozen, loaded from .pt)                                 #
# ------------------------------------------------------------------ #

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
        # self.agent.load_state_dict(checkpoint)
        # self.agent.eval()

        # handle both formats
        if "model_state_dict" in checkpoint:
            self.agent.load_state_dict(checkpoint["model_state_dict"])  # ← CleanRL format
        else:
            self.agent.load_state_dict(checkpoint)  # ← plain state dict
            
        self.agent.eval()

    def predict(self, obs, deterministic=True):
        obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = torch.argmax(self.agent.actor(obs_tensor), dim=-1)
        return action.cpu().numpy()[0], None

# ------------------------------------------------------------------ #
# LSTM Agent for evaluation                                           #
# ------------------------------------------------------------------ #

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
        obs_tensor  = torch.FloatTensor(obs).unsqueeze(0)  # (1, obs_dim)
        done_tensor = torch.FloatTensor([float(done)])     # (1,)
        with torch.no_grad():
            hidden, lstm_state = self.get_states(obs_tensor, lstm_state, done_tensor)
            if deterministic:
                action = torch.argmax(self.actor(hidden), dim=-1)
            else:
                action = Categorical(logits=self.actor(hidden)).sample()
        return action.cpu().numpy()[0], lstm_state


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         type=str,   required=True)
    parser.add_argument("--nav_model",     type=str,   default=None)
    parser.add_argument("--budget",        type=float, required=True)
    parser.add_argument("--fc",            type=float, required=True)
    parser.add_argument("--n_ep",          type=int,   default=1)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--fixed",         type=float, default=0.0)
    parser.add_argument("--output_dir",    type=str,   default="result_plots/nav_reward_study")
    parser.add_argument("--nav_rew_plots", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device("cpu")

    # load nav model
    nav_model_path = args.nav_model if args.nav_model else NAV_MODEL_PATH
    nav_model = NavModel(nav_model_path, device=device)
    print(f"Nav model loaded: {nav_model_path}")

    # load LSTM agent
    checkpoint = torch.load(args.model, map_location=device)
    agent = AgentEval(obs_dim=OBS_DIM, n_actions=N_ACTIONS, lstm_hidden_size=LSTM_HIDDEN_SIZE)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()
    print(f"Agent loaded: {args.model}")

    budget        = args.budget
    fc        = args.fc
    seed      = args.seed
    fixed_fps = args.fixed
    os.makedirs(args.output_dir, exist_ok=True)

    for ep in range(args.n_ep):

        env = gym.make("LunarLander_VarFramerate_SimplePadded",
                       frame_cost=fc, budget=budget, render_mode="rgb_array")
        env.unwrapped.navigation_model = nav_model
        env = TimeLimit(env, max_episode_steps=500)

        obs, _ = env.reset(seed=seed)
        done      = False
        truncated = False
        lstm_state = make_initial_lstm_state()

        touchdown_check = False
        touchdown_flag = False
        total_reward  = 0
        cum_nav_reward = 0
        cum_total_reward = 0
        step          = 0
        chosen_fps    = 0
        frames        = []
        nav_reward_per_timestep     = []
        cum_nav_reward_per_timestep = []
        total_reward_per_timestep   = []
        cum_total_reward_per_timestep = []
        height_per_timestep         = []
        vy_per_timestep             = []

        while not (done or truncated):

            if fixed_fps == 0.0:
                action, lstm_state = agent.predict(obs, lstm_state, done, deterministic=True)
            else:
                action = FPS_CHOICES.index(fixed_fps)

            obs, reward, done, truncated, info = env.step(action)
            true_obs = env.unwrapped.current_obs

            # reset lstm state if episode ended
            if done or truncated:
                lstm_state = make_initial_lstm_state()

            touchdown_check = (true_obs[6] or true_obs[7]) and not touchdown_flag
            if touchdown_check:
                touchdown_flag = True
                vy_at_touchdown = true_obs[3]
                print(f"TOUCHDOWN AT STEP {step} | Vy at Touchdown: {vy_at_touchdown:.2f}")

            height_per_timestep.append(true_obs[1])
            vy_per_timestep.append(true_obs[3])
            
            chosen_fps      = info["chosen_fps"]
            raw_nav_reward  = info["nav_reward"]
            raw_total_reward = info["reward"]
            consumed_frames = info["episode_frame_count"]
            nav_reward_per_timestep.append(raw_nav_reward)
            total_reward_per_timestep.append(raw_total_reward)
            cum_nav_reward += raw_nav_reward
            cum_nav_reward_per_timestep.append(cum_nav_reward)

            cum_total_reward += raw_total_reward
            cum_total_reward_per_timestep.append(cum_total_reward)

            frame = env.render()
            if frame is not None:
                frames.append(add_fps_overlay(frame, chosen_fps, step))

            print(f"Ep {ep+1} | Step {step} | Chosen FPS: {chosen_fps}")
            print(f"Height: {true_obs[1]:.2f} | Vertical Velocity: {true_obs[3]:.2f}")
            print(f"Consumed Frames: {consumed_frames}")
            #print(f"Raw Nav Reward: {raw_nav_reward}")
            step += 1

        env.close()

        gif_path = f"seed_{seed}_budget_{budget}_fc_{fc}_fps_demo.gif"
        save_gif(frames, os.path.join(args.output_dir, gif_path), gif_fps=GIF_FPS)

        if args.nav_rew_plots:
            fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(6, 1, figsize=(12, 24), sharex=True,
                                                                        gridspec_kw={"hspace": 0.05})
            steps = np.arange(len(nav_reward_per_timestep))

            # ── Height (top) ──────────────────────────────────
            ax0.plot(steps, height_per_timestep, color="seagreen", linewidth=1.5)
            ax0.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax0.set_ylabel("Height (y)")
            ax0.set_title(f"BUDGET={budget}  FC={fc}  seed={seed}")
            ax0.grid(True, alpha=0.3)

            # ── Vertical velocity ─────────────────────────────
            ax1.plot(steps, vy_per_timestep, color="darkorange", linewidth=1.5)
            ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax1.set_ylabel("Vertical Velocity (vy)")
            ax1.grid(True, alpha=0.3)

            # ── Raw nav reward ────────────────────────────────
            ax2.plot(steps, nav_reward_per_timestep, color="steelblue", linewidth=1.5)
            ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax2.set_ylabel("Nav Reward")
            ax2.grid(True, alpha=0.3)

            # ── Cumulative nav reward ─────────────────────────
            ax3.plot(steps, np.cumsum(nav_reward_per_timestep), color="firebrick", linewidth=1.5)
            ax3.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax3.set_ylabel("Cumulative Nav Reward")
            ax3.grid(True, alpha=0.3)

            # ── Raw total reward ──────────────────────────────
            ax4.plot(steps, total_reward_per_timestep, color="purple", linewidth=1.5)
            ax4.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax4.set_ylabel("Total Reward")
            ax4.grid(True, alpha=0.3)

            # ── Cumulative total reward ───────────────────────
            ax5.plot(steps, np.cumsum(total_reward_per_timestep), color="darkviolet", linewidth=1.5)
            ax5.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax5.set_ylabel("Cumulative Total Reward")
            ax5.set_xlabel("Timestep")
            ax5.grid(True, alpha=0.3)

            plt.tight_layout()
            if fixed_fps:
                plot_path = os.path.join(args.output_dir,
                    f"seed_{seed}_fixed_{fixed_fps}_budget_{budget}_fc_{fc}_nav_reward.png")
            else:
                plot_path = os.path.join(args.output_dir,
                    f"seed_{seed}_budget_{budget}_fc_{fc}_nav_reward.png")

            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Nav reward plot saved → {plot_path}")

        print(f"Episode {ep+1} | Total Reward: {cum_total_reward:.2f} | Total Nav Reward: {cum_nav_reward:.2f}")
        print("===============================")