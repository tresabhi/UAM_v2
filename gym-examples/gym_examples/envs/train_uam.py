# from stable_baselines3 import A2C
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
# Register your custom environment
from gymnasium.envs.registration import register
from uam_uav import Uam_Uav_Env

env = Uam_Uav_Env('Austin, Texas, USA', 8, 5, sleep_time=0.01)
#check_env(env)
obs, info = env.reset()


plt.ion()
fig, ax = env.render_init()
for _ in range(1500):
    action = env.action_space.sample()
    env.render(fig, ax)
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f'Action: {action}')
    print(f'Reward: {reward}')
    print(obs)

    if terminated or truncated :
        obs, info = env.reset()
    