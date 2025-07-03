
import os
import sys
import logging
import numpy as np
import matplotlib
# Try different backends in case TkAgg has issues
try:
    matplotlib.use('Agg')  # Fallback to Agg (no GUI required)
except Exception:
    pass  # Let matplotlib choose automatically
import matplotlib.pyplot as plt
import time
from datetime import datetime
import random
import signal
import traceback

# Stable-baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# Import your environment and reward function
from map_env_revised import MapEnv
from rewards_utils import _get_reward_simple, _get_reward_only_agent


def create_animation(model_path, seed=123, max_steps=500): #seed-101
    """Create an animation using the trained model"""
    logging.info("Creating animation for visualization")
    
    try:
        # Load the model
        model = PPO.load(model_path)
        
        # Create env with rendering
        env = MapEnv(
            number_of_uav=0,
            num_ORCA_uav=0,
            number_of_vertiport=10,
            location_name="Austin, Texas, USA", 
            airspace_tag_list=[], #("amenity", "hospital"), ("aeroway", "aerodrome")
            max_episode_steps=max_steps,
            seed=seed,
            obs_space_str="UAM_UAV",
            sorting_criteria="closest first",
            render_mode="human",  # Enable rendering
            max_uavs=4,
            max_vertiports=6
        )

        # Override the reward function
        env._get_reward = lambda: _get_reward_only_agent(env)
        
        # Run one episode
        obs, _ = env.reset(seed=seed)
        done = False
        steps = 0
        
        # Run until done or max steps
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            print("Inside create animation, STEP")
            print(env.agent.dynamics.is_learning)
            steps += 1
            done = terminated or truncated
        
        # Create animation
        logging.info(f"Creating animation after {steps} steps")
        ani = env.create_animation(steps)
        
        if ani:
            # Determine outcome for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if terminated and not info.get("collision", False):
                status = "success"
            elif info.get("collision", False):
                status = "collision"
            else:
                status = "timeout"
                
            # Save the animation
            animation_path = f"animations/flight_{status}_{steps}steps_{timestamp}"
            env.save_animation(ani, animation_path)
            logging.info(f"Animation saved to {animation_path}")
            return True
        else:
            logging.warning("Failed to create animation object")
            return False
    
    except Exception as e:
        logging.error(f"Animation error: {e}")
        traceback.print_exc()
        return False
    finally:
        if 'env' in locals():
            env.close()




model_path = 'models/ppo_map_env_20250703_152734.zip'
create_animation(model_path=model_path, max_steps=6000)
