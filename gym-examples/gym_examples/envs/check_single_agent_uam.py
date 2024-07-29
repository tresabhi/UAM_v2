"""
This script will check if the single agent uam environment follows stable-baselines3 standards.
Check that an environment follows Gym API. This is particularly useful when using a custom environment.
It also optionally check that the environment is compatible with Stable-Baselines.
Essentially, this script used sb3's check_env() to make sure all observation space, action space, and other necessary components of the RL environment follows guidelines set by openAI's gym and sb3.
"""

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env

# Register your custom environment
from gymnasium.envs.registration import register
from uam_uav import UamUavEnv

env = UamUavEnv("Austin, Texas, USA", 8, 5, sleep_time=0.01)
check_env(env)
