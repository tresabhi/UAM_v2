"""
Use this script to train a model using the custom UAM environment.
Before training, make sure to run check_uam.py to ensure the environment is in working condition. 
This check must be done anytime there is a change to the environment.

Once the check is complete you can proceed with training a model using algorithms from sb3
"""

# Local functions
from utils import *

# Required imports
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import matplotlib.pyplot as plt

# Register your custom environment
from gymnasium.envs.registration import register #this line () is not required - not a local package 'remove line when confirmed'
from uam_uav import UamUavEnv

import sys
import os
from datetime import datetime
import logging

def setup_logging(training_timesteps):
    """Setup logging to both file and console."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create unique log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/training_log_{training_timesteps}steps_{timestamp}.txt"
    
    # Setup logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger()

def create_env(max_steps=1000):
    """Create environment with consistent settings"""
    env = UamUavEnv(
        "Austin, Texas, USA",
        8,
        5,
        sleep_time=0.01,
        airspace_tag_list=[("building", "hospital"), ("aeroway", "aerodrome")],
    )
    env.max_episode_steps = max_steps #! why do i need to pass in the max steps like this, not pythonic, update/fix this 
    return env

def run_evaluation_episode(env, model, logger, max_steps):
    """Run a single evaluation episode with detailed logging."""
    # Clear any previous animation data
    env.clear_animation_data()
    
    # Reset environment
    obs, _ = env.reset()
    episode_reward = 0
    steps = 0
    
    logger.info("\nStarting evaluation episode...")
    logger.info(f"Initial observation: {obs}")
    
    ## Evaluation start ##
    while steps < max_steps:
        # Get action from model
        action, _states = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1
        
        # Detailed logging every 100 steps
        if steps % 100 == 0:
            logger.info(f"\nStep {steps}:")
            logger.info(f"Distance to goal: {info['distance_to_end_vertiport']:.2f}")
            logger.info(f"Current speed: {info.get('current_speed', 'N/A')}")
            logger.info(f"Current heading: {info.get('current_heading', 'N/A')}")
            
        # Log important flags
        if info.get('collision_static', False):
            logger.info("\nStatic collision detected!")
        if info.get('collision_dynamic', False):
            logger.info("\nDynamic collision detected!")
        
        if terminated:
            logger.info("\nGoal reached!")
            break
            
        if truncated:
            logger.info("\nEpisode truncated - Reason:")
            if info.get('collision_static', False):
                logger.info("- Static collision")
            if info.get('collision_dynamic', False):
                logger.info("- Dynamic collision")
            if info.get('timeout', False):
                logger.info("- Timeout")
            break
    ## Evaluation end ## 

    # Create animation if episode completed without collision 
    #! Might need to change the logic - render animation regardless of collision, add new functionality, even if collision is there 
    render_animation:bool = ( 
        (terminated and info.get('reached_goal', False)) or  # Reached goal
        (steps >= max_steps and not any([  # Reached max steps without collision
            info.get('collision_static', False),
            info.get('collision_dynamic', False)
        ]))
    )
    
    if render_animation and len(env.df) > 0:
        try:
            logger.info("\nCreating animation...")
            ani = env.create_animation(steps)
            if ani is not None:
                os.makedirs("animations", exist_ok=True)
                status = "success" if terminated else "timeout"
                animation_path = f"animations/flight_{status}_{steps}_steps.mp4"
                env.save_animation(ani, animation_path)
                logger.info(f"Animation saved to {animation_path}")
            else:
                logger.error("Failed to create animation object")
        except Exception as e:
            logger.error(f"Animation error: {e}", exc_info=True)
    
    return steps, episode_reward, terminated

def train_and_evaluate(training_timesteps=5000, max_episode_steps=500):
    """Train and evaluate with improved animation handling."""
    # Setup logging
    logger = setup_logging(training_timesteps)
    logger.info(f"Starting training with parameters:")
    logger.info(f"Training timesteps: {training_timesteps}")
    logger.info(f"Max episode steps: {max_episode_steps}")
    
    # Create environments
    env = create_env(max_steps=max_episode_steps)
    eval_env = create_env(max_steps=max_episode_steps)

    try:
        # Create and train model with optimized parameters for your reward function
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            clip_range=0.2,
            max_grad_norm=0.5,
            tensorboard_log="./logs/"
        )

        logger.info("Starting training...")
        model.learn(total_timesteps=training_timesteps, progress_bar=True)
        logger.info("Training complete!")

        # Save model
        os.makedirs("trained_models", exist_ok=True)
        model_path = f"trained_models/ppo_uam_{training_timesteps}"
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")

        # Run evaluation episodes
        max_eval_attempts = 5
        successful_episodes = 0
        
        for attempt in range(max_eval_attempts):
            logger.info(f"\nEvaluation attempt {attempt + 1}/{max_eval_attempts}")
            steps, episode_reward, success = run_evaluation_episode(
                eval_env, model, logger, max_episode_steps
            )
            
            if success:
                successful_episodes += 1
                logger.info(f"Successful evaluation on attempt {attempt + 1}")
            
            # Continue trying even after success to gather more data
            
        logger.info(f"\nCompleted evaluation with {successful_episodes} successful episodes")

    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
    finally:
        env.close()
        eval_env.close()

if __name__ == "__main__":
    # Start at 5000, 500 and scale accordingly by constant
    train_and_evaluate(25000, 2500)
