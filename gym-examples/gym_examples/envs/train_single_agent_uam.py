'''
Use this script to train a model using the custom UAM environment.
Before training, make sure to run check_uam.py to ensure the environment is in working condition. 
This check must be done anytime there is a change to the environment.

Once the check is complete you can proceed with training a model using algorithms from sb3
'''



# from stable_baselines3 import A2C
import gymnasium as gym
import matplotlib.pyplot as plt
# Register your custom environment
from gymnasium.envs.registration import register
from uam_uav import Uam_Uav_Env

env = Uam_Uav_Env('Austin, Texas, USA', 8, 5, sleep_time=0.01)
obs, info = env.reset()

# To have interactive plotting active 
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
    