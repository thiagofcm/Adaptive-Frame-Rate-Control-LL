import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from envs.lunar_lander_highest_fps import LunarLander_HighestFPS

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
    """Frozen nav model wrapper with predict() interface."""
    def __init__(self, model_path, device):
        self.device = device
        checkpoint = torch.load(model_path, map_location=device)
        self.agent = NavAgent().to(device)
        # handle both formats
        if "model_state_dict" in checkpoint:
            self.agent.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.agent.load_state_dict(checkpoint)
        self.agent.eval()

    def predict(self, obs, deterministic=True):
        obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.agent.actor(obs_tensor)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return action.cpu().numpy()[0], None

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def predict(self, obs, deterministic=True):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=next(self.parameters()).device)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(obs_tensor)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = Categorical(logits=logits).sample()
        return action.cpu().numpy()[0], None
