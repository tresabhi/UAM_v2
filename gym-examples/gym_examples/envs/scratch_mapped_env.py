import numpy as np 
import math
import matplotlib.pyplot as plt

import random
import shapely
from map_renderer import MapRenderer
from map_logging_loader import MapLoader
from map_env_revised import MapEnv
from rewards_utils import _get_reward_only_agent
from utils_data_transform import normalize_minus_one_one, normalize_zero_one





def simple_controller(current_position:shapely.Point, final_position:shapely.Point, current_heading:float):
    
    acceleration = np.random.uniform(0,1)
    final_heading = math.atan2(final_position.y - current_position.y, 
                                final_position.x - current_position.x)
    
    heading_change_rad = final_heading - current_heading
    
    heading_change = normalize_minus_one_one(heading_change_rad, -math.pi, math.pi)

    action = np.array([acceleration, heading_change])
    
    return action









def simple_reward(self):
    return 0.0

number_orca_agents = 0
number_uav = 0
number_of_vp = 8
episodes = 1
max_steps_per_episode = 30
render = False
save_animation = False
env_seed = 42
episode_seeds = 60
mp4_only = False


MapEnv._get_reward = _get_reward_only_agent

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




# print('The following is a sample of observation space:')
# print(env.observation_space.sample())

# print('sample action:')
# sample_action = env.action_space.sample()
# print(sample_action)
# sa_type = type(sample_action)
# print('Type:', sa_type)

# print(sample_action.shape)

# env.reset()

# print('Raw obs: ')
# print(env.agent.get_obs()) # <- this is raw obs 






for episode in range(episodes):
    obs, info = env.reset(seed=episode_seeds)

    print('Observation: ')
    print(obs)

    print('Info:')
    print(info)

    for step in range(max_steps_per_episode):
        # action = env.action_space.sample()
        action = simple_controller(env.agent.current_position, env.agent.end, env.agent.current_heading)

        obs, reward, terminated, truncated, info = env.step(action)
        

        print('Observation: ')
        print(obs)

        print('Reward:')
        print(reward)

        print('Info:')
        print(info)

 
        if terminated:
            print('Terminated')
        
        if truncated:
            if 'collision' in info:
                print(f"Collision detected at step {step}!")
                collision_detected = True
            elif 'timeout' in info and info['timeout']:
                print(f"Episode timeout after {step} steps!")
            break
        

















