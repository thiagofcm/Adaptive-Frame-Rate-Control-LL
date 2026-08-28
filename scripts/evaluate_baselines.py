"""
Stage 1: interpretable, non-learned sensing baselines for Lunar Lander.

Mirrors scripts/evaluate_lunar_lander_fps.py's evaluation structure as closely as
possible -- same environment, same frozen navigation controller, same FPS action
mechanism/obs_interval behavior, same steps.csv/episodes.csv/trajectory.npy/config.json/
summary.csv logic. The only difference is how the FPS action is selected: instead of a
--fixed-fps int or a learned adaptive-sensing policy, one of these hand-designed
strategies chooses it every step:

    height  -- observed altitude (y) thresholded into one of [1, 5, 10, 25, 50] Hz
    random  -- uniform random draw from [1, 5, 10, 25, 50] Hz, reproducibly seeded

Fairness: each strategy function receives ONLY the same (possibly stale) 11-dim
observation the adaptive policy itself receives -- never env.unwrapped.current_obs or
any other ground-truth state. After the first touchdown, every strategy requests 1 Hz;
that override is applied by the evaluator (which already tracks touchdown_flag from
ground truth for success/logging, exactly as the existing evaluator does), never inside
the strategy function itself.

The kinematic risk heuristic and threshold tuning/search are deliberately out of scope
for this stage.

Run from the repo root with PYTHONPATH=<repo root>:

    # run every named height schedule (HEIGHT_SCHEDULES) in one invocation
    PYTHONPATH=. python scripts/evaluate_baselines.py --strategy height --n-episodes 100

    # or just one
    PYTHONPATH=. python scripts/evaluate_baselines.py --strategy height \\
        --schedule C --n-episodes 100

    PYTHONPATH=. python scripts/evaluate_baselines.py --strategy random --n-episodes 100
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
from gymnasium.wrappers import TimeLimit

import envs.lunar_lander_var_fps  # noqa: F401  (registers LunarLander_VarFramerate)

# =========================
# Constants (matching scripts/evaluate_lunar_lander_fps.py exactly)
# =========================
RUN_SEED = 42

FPS_TO_ACTION = {1: 0, 5: 1, 10: 2, 25: 3, 50: 4}
FPS_CHOICES = [1, 5, 10, 25, 50]

# Named height-band schedules -- the only way --strategy height is configured (see
# height_schedule_fps). --schedule picks one; omit it to run all of them in one
# invocation (see main()).

HEIGHT_SCHEDULES = {
    "S001": [5, 5, 5, 5],
    "S002": [5, 5, 5, 10],
    "S003": [5, 5, 5, 25],
    "S004": [5, 5, 5, 50],
    "S005": [5, 5, 10, 5],
    "S006": [5, 5, 10, 10],
    "S007": [5, 5, 10, 25],
    "S008": [5, 5, 10, 50],
    "S009": [5, 5, 25, 5],
    "S010": [5, 5, 25, 10],
    "S011": [5, 5, 25, 25],
    "S012": [5, 5, 25, 50],
    "S013": [5, 5, 50, 5],
    "S014": [5, 5, 50, 10],
    "S015": [5, 5, 50, 25],
    "S016": [5, 5, 50, 50],
    "S017": [5, 10, 5, 5],
    "S018": [5, 10, 5, 10],
    "S019": [5, 10, 5, 25],
    "S020": [5, 10, 5, 50],
    "S021": [5, 10, 10, 5],
    "S022": [5, 10, 10, 10],
    "S023": [5, 10, 10, 25],
    "S024": [5, 10, 10, 50],
    "S025": [5, 10, 25, 5],
    "S026": [5, 10, 25, 10],
    "S027": [5, 10, 25, 25],
    "S028": [5, 10, 25, 50],
    "S029": [5, 10, 50, 5],
    "S030": [5, 10, 50, 10],
    "S031": [5, 10, 50, 25],
    "S032": [5, 10, 50, 50],
    "S033": [5, 25, 5, 5],
    "S034": [5, 25, 5, 10],
    "S035": [5, 25, 5, 25],
    "S036": [5, 25, 5, 50],
    "S037": [5, 25, 10, 5],
    "S038": [5, 25, 10, 10],
    "S039": [5, 25, 10, 25],
    "S040": [5, 25, 10, 50],
    "S041": [5, 25, 25, 5],
    "S042": [5, 25, 25, 10],
    "S043": [5, 25, 25, 25],
    "S044": [5, 25, 25, 50],
    "S045": [5, 25, 50, 5],
    "S046": [5, 25, 50, 10],
    "S047": [5, 25, 50, 25],
    "S048": [5, 25, 50, 50],
    "S049": [5, 50, 5, 5],
    "S050": [5, 50, 5, 10],
    "S051": [5, 50, 5, 25],
    "S052": [5, 50, 5, 50],
    "S053": [5, 50, 10, 5],
    "S054": [5, 50, 10, 10],
    "S055": [5, 50, 10, 25],
    "S056": [5, 50, 10, 50],
    "S057": [5, 50, 25, 5],
    "S058": [5, 50, 25, 10],
    "S059": [5, 50, 25, 25],
    "S060": [5, 50, 25, 50],
    "S061": [5, 50, 50, 5],
    "S062": [5, 50, 50, 10],
    "S063": [5, 50, 50, 25],
    "S064": [5, 50, 50, 50],
    "S065": [10, 5, 5, 5],
    "S066": [10, 5, 5, 10],
    "S067": [10, 5, 5, 25],
    "S068": [10, 5, 5, 50],
    "S069": [10, 5, 10, 5],
    "S070": [10, 5, 10, 10],
    "S071": [10, 5, 10, 25],
    "S072": [10, 5, 10, 50],
    "S073": [10, 5, 25, 5],
    "S074": [10, 5, 25, 10],
    "S075": [10, 5, 25, 25],
    "S076": [10, 5, 25, 50],
    "S077": [10, 5, 50, 5],
    "S078": [10, 5, 50, 10],
    "S079": [10, 5, 50, 25],
    "S080": [10, 5, 50, 50],
    "S081": [10, 10, 5, 5],
    "S082": [10, 10, 5, 10],
    "S083": [10, 10, 5, 25],
    "S084": [10, 10, 5, 50],
    "S085": [10, 10, 10, 5],
    "S086": [10, 10, 10, 10],
    "S087": [10, 10, 10, 25],
    "S088": [10, 10, 10, 50],
    "S089": [10, 10, 25, 5],
    "S090": [10, 10, 25, 10],
    "S091": [10, 10, 25, 25],
    "S092": [10, 10, 25, 50],
    "S093": [10, 10, 50, 5],
    "S094": [10, 10, 50, 10],
    "S095": [10, 10, 50, 25],
    "S096": [10, 10, 50, 50],
    "S097": [10, 25, 5, 5],
    "S098": [10, 25, 5, 10],
    "S099": [10, 25, 5, 25],
    "S100": [10, 25, 5, 50],
    "S101": [10, 25, 10, 5],
    "S102": [10, 25, 10, 10],
    "S103": [10, 25, 10, 25],
    "S104": [10, 25, 10, 50],
    "S105": [10, 25, 25, 5],
    "S106": [10, 25, 25, 10],
    "S107": [10, 25, 25, 25],
    "S108": [10, 25, 25, 50],
    "S109": [10, 25, 50, 5],
    "S110": [10, 25, 50, 10],
    "S111": [10, 25, 50, 25],
    "S112": [10, 25, 50, 50],
    "S113": [10, 50, 5, 5],
    "S114": [10, 50, 5, 10],
    "S115": [10, 50, 5, 25],
    "S116": [10, 50, 5, 50],
    "S117": [10, 50, 10, 5],
    "S118": [10, 50, 10, 10],
    "S119": [10, 50, 10, 25],
    "S120": [10, 50, 10, 50],
    "S121": [10, 50, 25, 5],
    "S122": [10, 50, 25, 10],
    "S123": [10, 50, 25, 25],
    "S124": [10, 50, 25, 50],
    "S125": [10, 50, 50, 5],
    "S126": [10, 50, 50, 10],
    "S127": [10, 50, 50, 25],
    "S128": [10, 50, 50, 50],
    "S129": [25, 5, 5, 5],
    "S130": [25, 5, 5, 10],
    "S131": [25, 5, 5, 25],
    "S132": [25, 5, 5, 50],
    "S133": [25, 5, 10, 5],
    "S134": [25, 5, 10, 10],
    "S135": [25, 5, 10, 25],
    "S136": [25, 5, 10, 50],
    "S137": [25, 5, 25, 5],
    "S138": [25, 5, 25, 10],
    "S139": [25, 5, 25, 25],
    "S140": [25, 5, 25, 50],
    "S141": [25, 5, 50, 5],
    "S142": [25, 5, 50, 10],
    "S143": [25, 5, 50, 25],
    "S144": [25, 5, 50, 50],
    "S145": [25, 10, 5, 5],
    "S146": [25, 10, 5, 10],
    "S147": [25, 10, 5, 25],
    "S148": [25, 10, 5, 50],
    "S149": [25, 10, 10, 5],
    "S150": [25, 10, 10, 10],
    "S151": [25, 10, 10, 25],
    "S152": [25, 10, 10, 50],
    "S153": [25, 10, 25, 5],
    "S154": [25, 10, 25, 10],
    "S155": [25, 10, 25, 25],
    "S156": [25, 10, 25, 50],
    "S157": [25, 10, 50, 5],
    "S158": [25, 10, 50, 10],
    "S159": [25, 10, 50, 25],
    "S160": [25, 10, 50, 50],
    "S161": [25, 25, 5, 5],
    "S162": [25, 25, 5, 10],
    "S163": [25, 25, 5, 25],
    "S164": [25, 25, 5, 50],
    "S165": [25, 25, 10, 5],
    "S166": [25, 25, 10, 10],
    "S167": [25, 25, 10, 25],
    "S168": [25, 25, 10, 50],
    "S169": [25, 25, 25, 5],
    "S170": [25, 25, 25, 10],
    "S171": [25, 25, 25, 25],
    "S172": [25, 25, 25, 50],
    "S173": [25, 25, 50, 5],
    "S174": [25, 25, 50, 10],
    "S175": [25, 25, 50, 25],
    "S176": [25, 25, 50, 50],
    "S177": [25, 50, 5, 5],
    "S178": [25, 50, 5, 10],
    "S179": [25, 50, 5, 25],
    "S180": [25, 50, 5, 50],
    "S181": [25, 50, 10, 5],
    "S182": [25, 50, 10, 10],
    "S183": [25, 50, 10, 25],
    "S184": [25, 50, 10, 50],
    "S185": [25, 50, 25, 5],
    "S186": [25, 50, 25, 10],
    "S187": [25, 50, 25, 25],
    "S188": [25, 50, 25, 50],
    "S189": [25, 50, 50, 5],
    "S190": [25, 50, 50, 10],
    "S191": [25, 50, 50, 25],
    "S192": [25, 50, 50, 50],
    "S193": [50, 5, 5, 5],
    "S194": [50, 5, 5, 10],
    "S195": [50, 5, 5, 25],
    "S196": [50, 5, 5, 50],
    "S197": [50, 5, 10, 5],
    "S198": [50, 5, 10, 10],
    "S199": [50, 5, 10, 25],
    "S200": [50, 5, 10, 50],
    "S201": [50, 5, 25, 5],
    "S202": [50, 5, 25, 10],
    "S203": [50, 5, 25, 25],
    "S204": [50, 5, 25, 50],
    "S205": [50, 5, 50, 5],
    "S206": [50, 5, 50, 10],
    "S207": [50, 5, 50, 25],
    "S208": [50, 5, 50, 50],
    "S209": [50, 10, 5, 5],
    "S210": [50, 10, 5, 10],
    "S211": [50, 10, 5, 25],
    "S212": [50, 10, 5, 50],
    "S213": [50, 10, 10, 5],
    "S214": [50, 10, 10, 10],
    "S215": [50, 10, 10, 25],
    "S216": [50, 10, 10, 50],
    "S217": [50, 10, 25, 5],
    "S218": [50, 10, 25, 10],
    "S219": [50, 10, 25, 25],
    "S220": [50, 10, 25, 50],
    "S221": [50, 10, 50, 5],
    "S222": [50, 10, 50, 10],
    "S223": [50, 10, 50, 25],
    "S224": [50, 10, 50, 50],
    "S225": [50, 25, 5, 5],
    "S226": [50, 25, 5, 10],
    "S227": [50, 25, 5, 25],
    "S228": [50, 25, 5, 50],
    "S229": [50, 25, 10, 5],
    "S230": [50, 25, 10, 10],
    "S231": [50, 25, 10, 25],
    "S232": [50, 25, 10, 50],
    "S233": [50, 25, 25, 5],
    "S234": [50, 25, 25, 10],
    "S235": [50, 25, 25, 25],
    "S236": [50, 25, 25, 50],
    "S237": [50, 25, 50, 5],
    "S238": [50, 25, 50, 10],
    "S239": [50, 25, 50, 25],
    "S240": [50, 25, 50, 50],
    "S241": [50, 50, 5, 5],
    "S242": [50, 50, 5, 10],
    "S243": [50, 50, 5, 25],
    "S244": [50, 50, 5, 50],
    "S245": [50, 50, 10, 5],
    "S246": [50, 50, 10, 10],
    "S247": [50, 50, 10, 25],
    "S248": [50, 50, 10, 50],
    "S249": [50, 50, 25, 5],
    "S250": [50, 50, 25, 10],
    "S251": [50, 50, 25, 25],
    "S252": [50, 50, 25, 50],
    "S253": [50, 50, 50, 5],
    "S254": [50, 50, 50, 10],
    "S255": [50, 50, 50, 25],
    "S256": [50, 50, 50, 50],
    "X1": [50, 1, 50, 50],
    "X2": [50, 1, 50, 50],
    "X3": [50, 50, 1, 50],
    "X4": [50, 50, 50, 1],
}


# An empirically selected reference height based on the observed starting-altitude
# range (eval/fixed_25Hz sampling this session showed max observed y ~= 1.52 across 50
# episodes) -- not a uniquely correct or theoretically derived value, just a reasonable,
# consistent choice reused from that prior finding.
HEIGHT_REFERENCE = 1.5

NAV_MODEL_PATH_DEFAULT = "experiments/navigation/runs/LunarLander-v3__ppo__1__1779191150/model.pt"

EVAL_ROOT = "eval/vy_exceed_0.5/heuristic"

# Baselines measure landing performance vs. raw sensing consumption -- not the adaptive
# policy's training-time reward shaping. frame_cost=0 means no per-frame reward penalty;
# budget=inf means the budget-overrun penalty (_physics_step's
# `episode_frame_count > budget` check) can never fire. Neither affects any FPS
# *decision* -- only the reward magnitude that gets logged -- so neither is CLI-exposed,
# tunable, or part of the output folder name. budget=0.0 (the env constructor's own
# default) would instead be actively harmful here: episode_frame_count starts at 1, so
# `episode_frame_count > budget` would be True from the first step of every episode,
# firing the -100 budget-overrun penalty almost immediately.
NEUTRAL_FRAME_COST = 0.0
NEUTRAL_BUDGET = float("inf")

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
# Model classes -- copied verbatim from scripts/evaluate_lunar_lander_fps.py so this
# evaluator preserves the exact frozen-controller architecture. No AgentEval/LSTM here:
# no learned sensing policy is involved in this stage.
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


# =========================
# Sensing strategies -- pure functions of the observation the adaptive policy would
# also see (index 0/1/2/3 = x/y/vx/vy; confirmed against envs/lunar_lander_var_fps.py's
# _get_augmented_obs/_physics_step). NEVER read env.unwrapped.current_obs or any other
# ground-truth state here -- that would defeat the fairness comparison against the
# learned adaptive policy, which only ever sees this same observation.
# =========================

def height_schedule_fps(observation, schedule):
    """Fairness contract: observation[1]=y only, no ground truth (never
    env.unwrapped.current_obs). Divides [0, HEIGHT_REFERENCE] into four equal 25% bands
    and applies one of the four named FPS schedules (see HEIGHT_SCHEDULES)."""
    frac = observation[1] / HEIGHT_REFERENCE
    if frac > 0.75:
        return schedule[0]
    if frac > 0.5:
        return schedule[1]
    if frac > 0.25:
        return schedule[2]
    return schedule[3]


def random_fps(rng):
    return int(rng.choice(FPS_CHOICES))


# =========================
# Evaluate one episode -- adapted from scripts/evaluate_lunar_lander_fps.py's
# run_episode(). Same touchdown/reward/CSV-row-building logic verbatim; the only change
# is the action-selection line (fixed_fps/model dispatch -> strategy_fn + touchdown
# override).
# =========================

def run_episode(env, strategy_fn, episode_dir, seed):
    """Run one full episode, write that episode's steps.csv + trajectory .npy, and
    return one summary dict (an episodes.csv row)."""

    observation, info = env.reset(seed=seed)

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
    touchdown_flag = False
    touchdown_vy = None
    landed_in_flags = False
    outside_flags_after_landing = False
    went_up_after = False
    exceed_vy_vel = False
    post_touchdown_airborne_steps = 0
    terminated, truncated = False, False
    touchdown_steps = 0

    while not (terminated or truncated):

        # --- the ONLY difference vs. the fixed/adaptive evaluator: how the FPS action
        # is chosen. strategy_fn never sees touchdown_flag or any ground-truth state --
        # only `observation`, the same (possibly stale) array the adaptive policy would
        # receive. The touchdown override is applied here, after the strategy call, using
        # touchdown_flag as it stood at the end of the *previous* step -- no future
        # information leaks into the current action.
        raw_fps = strategy_fn(observation)
        if touchdown_flag:
            touchdown_steps += 1

        fps = 1 if touchdown_steps > 5 else raw_fps
        action = FPS_TO_ACTION[fps]

        # ALTERNATIVE WAY THAT CAUSES BOUNCES:
        #fps = 1 if touchdown_flag else raw_fps 
        # action = FPS_TO_ACTION[fps]
        # ------------------------------------------------------

        observation, reward, terminated, truncated, info = env.step(action)
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

        # A real bounce can lift off gradually (both legs -> one leg -> no legs), not
        # just as an immediate both-grounded-to-both-off transition, so this counts
        # consecutive fully-airborne steps after touchdown instead: 1 is tolerated as
        # possible contact jitter, 2+ means the lander has genuinely left the ground.
        # went_up_after is only ever set True here, never reset, so it stays True for
        # the rest of the episode once tripped.
        if touchdown_flag:
            if not leg1 and not leg2:
                post_touchdown_airborne_steps += 1
            else:
                post_touchdown_airborne_steps = 0

            if post_touchdown_airborne_steps >= 5:
                went_up_after = True

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
            "n_fresh_observations_before_touchdown": n_fresh_observations_before_touchdown,
            "n_fresh_observations_after_touchdown": n_fresh_observations_after_touchdown,
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
        "n_fresh_observations_before_touchdown": n_fresh_observations_before_touchdown,
        "n_fresh_observations_after_touchdown": n_fresh_observations_after_touchdown,
        "fresh_observation_ratio": n_fresh_observations / step if step else 0.0,
        "fresh_observation_ratio_b4_td": n_fresh_observations_before_touchdown / touchdown_step if touchdown_step is not None and touchdown_step > 0 else 0.0,
        "mean_fps": float(np.mean(fps_trace)) if fps_trace else "",
    }


def summarize_results():
    """Read every */episodes.csv under eval/ and combine them into one comparison
    table -- copied verbatim from scripts/evaluate_lunar_lander_fps.py so baseline runs
    join the same aggregate eval/summary.csv as adaptive/fixed runs. Full rebuild every
    time, registered to run on interpreter exit."""
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
        n_fresh_before_touchdown = np.array([float(r["n_fresh_observations_before_touchdown"]) for r in rows])
        n_fresh_after_touchdown = np.array([float(r["n_fresh_observations_after_touchdown"]) for r in rows])
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
            "mean_n_fresh_observations_before_touchdown": float(n_fresh_before_touchdown.mean()),
            "std_n_fresh_observations_before_touchdown": float(n_fresh_before_touchdown.std()),
            "mean_n_fresh_observations_after_touchdown": float(n_fresh_after_touchdown.mean()),
            "std_n_fresh_observations_after_touchdown": float(n_fresh_after_touchdown.std()),
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


def run_strategy(env, strategy_label, strategy_fn, strategy_params, out_dir, args):
    """Run args.n_episodes episodes for one strategy configuration (one height schedule,
    or random) and write its episodes.csv/config.json (+ configuration.txt for height
    schedules) into out_dir."""
    episode_rows = []
    for ep_index in range(args.n_episodes):
        episode_dir = f"{out_dir}/lap_{ep_index:02d}"
        seed = RUN_SEED + ep_index

        if strategy_label == "random":
            rng = np.random.default_rng(seed)
            episode_strategy_fn = lambda observation, rng=rng: random_fps(rng)
        else:
            episode_strategy_fn = strategy_fn

        row = run_episode(env, episode_strategy_fn, episode_dir, seed)
        row["episode_index"] = ep_index
        episode_rows.append(row)

        print(
            f"[{strategy_label}] Episode {ep_index}: "
            f"success={row['success']} "
            f"episode_length={row['episode_length']} "
            f"adaptive_reward={row['adaptive_reward']:.3f} "
            f"nav_reward={row['nav_reward']:.3f} "
            f"mean_fps={row['mean_fps']:.2f} "
            f"fresh_obs_ratio_b4_td={row['fresh_observation_ratio_b4_td']:.2f} "
            f"fresh_obs_ratio={row['fresh_observation_ratio']:.2f}"
        )

    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/episodes.csv", 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=EPISODE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(episode_rows)
    print(f"Episode results written to: {out_dir}/episodes.csv")

    config = {
        "mode": strategy_label,
        "strategy_params": strategy_params,
        "nav_model": args.nav_model,
        "frame_cost": NEUTRAL_FRAME_COST,
        "budget": "inf",  # informational only -- see NEUTRAL_BUDGET; "inf" (string) so
                          # this file stays strict-JSON-parseable, unlike a raw float('inf')
        "episodes": args.n_episodes,
        "initial seed": RUN_SEED,
        "max_episode_steps": 500,
        "device": "cpu",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(f"{out_dir}/config.json", "w") as file:
        json.dump(config, file, indent=2)

    # Human-readable companion to config.json -- only for named height schedules.
    if strategy_label.startswith("height_"):
        schedule_name = strategy_params["schedule_name"]
        schedule = strategy_params["schedule"]
        configuration_text = (
            "Strategy: Height-based sensing\n"
            f"Schedule: {schedule_name}\n"
            f"FPS schedule: {schedule} Hz\n"
            "\n"
            "Height bands:\n"
            f"  75%-100% of reference height -> {schedule[0]} Hz\n"
            f"  50%-75%                      -> {schedule[1]} Hz\n"
            f"  25%-50%                      -> {schedule[2]} Hz\n"
            f"  0%-25%                       -> {schedule[3]} Hz\n"
            "\n"
            f"Reference height: {HEIGHT_REFERENCE}\n"
            "\n"
            "After first touchdown: request 1 Hz\n"
            "\n"
            "Observation policy:\n"
            "  FPS decisions use the same potentially stale observation\n"
            "  available to the sensing policy. No ground-truth state is\n"
            "  used for FPS selection.\n"
            "\n"
            "Evaluation:\n"
            f"  RUN_SEED: {RUN_SEED}\n"
            f"  Episodes: {args.n_episodes}\n"
            "\n"
            "Neutral environment parameters:\n"
            f"  frame_cost: {NEUTRAL_FRAME_COST}\n"
            "  budget: inf\n"
        )
        with open(f"{out_dir}/configuration.txt", "w") as file:
            file.write(configuration_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["height", "random"], required=True)
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--nav-model", type=str, default=NAV_MODEL_PATH_DEFAULT,
                         help="Path to the frozen navigation/landing controller checkpoint.")
    parser.add_argument("--output-name", type=str, default=None,
                         help="Policy-run folder name under eval/. Ignored when --strategy "
                              "height runs every schedule in one invocation (no --schedule "
                              "given), to avoid every schedule colliding on the same folder.")

    # --strategy height only: run this one named schedule; omit to run every schedule in
    # HEIGHT_SCHEDULES, one after another, in the same invocation.
    parser.add_argument("--schedule", choices=list(HEIGHT_SCHEDULES), default=None)

    args = parser.parse_args()

    # Calls summarize results at the end of the script
    atexit.register(summarize_results)

    # Creates LunarLander Env -- frame_cost/budget are internal neutral values only
    # (see NEUTRAL_FRAME_COST/NEUTRAL_BUDGET above), never CLI-exposed for baselines.
    nav_model = NavModel(args.nav_model, device=torch.device("cpu"))
    env = gym.make("LunarLander_VarFramerate", frame_cost=NEUTRAL_FRAME_COST, budget=NEUTRAL_BUDGET)
    env.unwrapped.navigation_model = nav_model
    env = TimeLimit(env, max_episode_steps=500)

    if args.strategy == "height":
        schedule_names = [args.schedule] if args.schedule is not None else list(HEIGHT_SCHEDULES)
        running_all = len(schedule_names) > 1
        for schedule_name in schedule_names:
            schedule = HEIGHT_SCHEDULES[schedule_name]
            strategy_params = {
                "schedule_name": schedule_name,
                "schedule": schedule,
                "reference_height": HEIGHT_REFERENCE,
            }
            strategy_fn = lambda observation, schedule=schedule: height_schedule_fps(observation, schedule)
            strategy_label = f"height_{schedule_name}"
            default_name = f"height_{schedule_name}"
            # --output-name would collide across schedules when running all of them, so
            # it's only honored for a single-schedule (or random) run.
            out_dir = f"{EVAL_ROOT}/{default_name if running_all else (args.output_name or default_name)}"
            run_strategy(env, strategy_label, strategy_fn, strategy_params, out_dir, args)
    else:
        out_dir = f"{EVAL_ROOT}/{args.output_name or 'random'}"
        run_strategy(env, "random", None, {}, out_dir, args)

    env.close()


if __name__ == "__main__":
    main()
