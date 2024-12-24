#local functions
from utils import *

# from stable_baselines3 import A2C
import gymnasium as gym
import matplotlib.pyplot as plt
import stable_baselines3 as sb3 

# Register your custom environment
from gymnasium.envs.registration import register
from uam_uav import UamUavEnv

env = UamUavEnv("Austin, Texas, USA", 8, 5, sleep_time=0.01, airspace_tag_list=[("building", "hospital"),("aeroway", "aerodrome")])
obs, info = env.reset()

# For interactive plotting - to see rendering of airspace during training
# plt.ion()
# fig, ax = env.render_init()
training_time_step = 50

for i in range(training_time_step):
    action = env.action_space.sample()
    #env.render(fig, ax)
    obs, reward, terminated, truncated, info = env.step(action)

    # For debugging intruder uav detection
    # if obs["intruder_detected"]:
    #     print(f"Intruder ID: {obs['intruder_id']}")
    #     print(f"Distance to intruder: {obs['distance_to_intruder']}")
    #     print(f"Relative heading with intruder: {obs['relative_heading_intruder']}")
    #     print(f"Intruder heading: {obs['intruder_current_heading']}")

    # # For debugging - speed
    # auto_uav_speed = env.auto_uav.current_speed
    # auto_uav_heading = env.auto_uav.current_heading_deg
    # auto_uav_ref_final_heading = env.auto_uav.current_ref_final_heading_deg
    # print(f"Action: {action}")
    # print(f"Current speed: {auto_uav_speed}")
    # print(f"Current heading: {auto_uav_heading}")
    # print(f"Reference final heading: {auto_uav_ref_final_heading}")
    # print(f"Reward: {reward}")
    # distance_to_end_vp = info["distance_to_end_vertiport"]
    # print(f"Distance to target: {distance_to_end_vp}")
    # # print(obs)

    # if terminated or truncated:
    #     obs, info = env.reset()


ani = env.create_animation(training_time_step)
env.save_animation(ani,'train_uav')







