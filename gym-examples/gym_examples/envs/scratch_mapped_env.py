import numpy as np 
import matplotlib.pyplot as plt

import random
from map_renderer import MapRenderer
from map_logging_loader import MapLoader
from map_env_revised import MapEnv



def simple_reward(self):
    return 0.0

number_orca_agents = 3
number_uav = 2
number_of_vp = 8
episodes = 1
max_steps_per_episode = 2
render = False
save_animation = False
env_seed = 42
episode_seeds = 60
mp4_only = False


MapEnv._get_reward = simple_reward

env = MapEnv(
            number_of_uav= number_uav,
            num_ORCA_uav=number_orca_agents,
            number_of_vertiport= number_of_vp,
            location_name="Austin, Texas, USA",
            airspace_tag_list=[], #("amenity", "hospital"), ("aeroway", "aerodrome")
            max_episode_steps=max_steps_per_episode,
            seed=env_seed,  # Use the specified environment seed
            obs_space_str= "UAM_UAV", #"UAV_5_intruders",  # "LSTM-A2C",
            sorting_criteria= 'closest first',#None, # "closest first",
            render_mode="human" if render else None,
            max_uavs=100, #set these as some hyperparameters 
            max_vertiports=150, #set these as some hyperparameters 
        )

print('The following is a sample of observation space:')
print(env.observation_space.sample())

print()
for episode in range(episodes):
    obs, info = env.reset(seed=episode_seeds)

    for step in range(max_steps_per_episode):
        action = env.action_space.sample()

        next_obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            print('Terminated')
        
        if truncated:
            if 'collision' in info:
                print(f"Collision detected at step {step}!")
                collision_detected = True
            elif 'timeout' in info and info['timeout']:
                print(f"Episode timeout after {step} steps!")
            break
        