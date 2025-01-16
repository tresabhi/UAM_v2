import gymnasium as gym
from stable_baselines3.common.env_checker import check_env

# Register your custom environment
from gymnasium.envs.registration import register
from simple_env import SimpleEnv

env = SimpleEnv()
check_env(env)