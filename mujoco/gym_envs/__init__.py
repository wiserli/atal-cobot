"""WISERLI-ATAL-C1.1 Farama Gymnasium Reinforcement Learning Environments."""

from gymnasium.envs.registration import register

from .base_env import WiserliAtalBaseEnv
from .reach_env import WiserliAtalReachEnv
from .pick_and_place_env import WiserliAtalPickAndPlaceEnv
from .cable_env import WiserliAtalCableEnv

# Register Task-Space (Cartesian) Environments
register(
    id="WiserliAtalReach-v0",
    entry_point="gym_envs.reach_env:WiserliAtalReachEnv",
    max_episode_steps=150,
    kwargs={"action_mode": "cartesian"},
)

register(
    id="WiserliAtalPickAndPlace-v0",
    entry_point="gym_envs.pick_and_place_env:WiserliAtalPickAndPlaceEnv",
    max_episode_steps=200,
    kwargs={"action_mode": "cartesian"},
)

register(
    id="WiserliAtalCable-v0",
    entry_point="gym_envs.cable_env:WiserliAtalCableEnv",
    max_episode_steps=150,
    kwargs={"action_mode": "cartesian"},
)

# Register Joint-Space Environments
register(
    id="WiserliAtalReachJoint-v0",
    entry_point="gym_envs.reach_env:WiserliAtalReachEnv",
    max_episode_steps=150,
    kwargs={"action_mode": "joint_position"},
)

register(
    id="WiserliAtalPickAndPlaceJoint-v0",
    entry_point="gym_envs.pick_and_place_env:WiserliAtalPickAndPlaceEnv",
    max_episode_steps=200,
    kwargs={"action_mode": "joint_position"},
)

register(
    id="WiserliAtalCableJoint-v0",
    entry_point="gym_envs.cable_env:WiserliAtalCableEnv",
    max_episode_steps=150,
    kwargs={"action_mode": "joint_position"},
)

__all__ = [
    "WiserliAtalBaseEnv",
    "WiserliAtalReachEnv",
    "WiserliAtalPickAndPlaceEnv",
    "WiserliAtalCableEnv",
]
