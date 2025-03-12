import numpy as np
import matplotlib
# Try different backends in case TkAgg has issues
try:
    matplotlib.use('TkAgg')  # Try TkAgg first
except Exception:
    try:
        matplotlib.use('Agg')  # Fallback to Agg (no GUI required)
    except Exception:
        pass  # Let matplotlib choose automatically

import matplotlib.pyplot as plt
from map_env_revised import MapEnv
import time
import os
import traceback
import sys
import shutil
import random
from map_logging_loader import MapLoader

# Ensure no global randomness affects our tests
random.seed(0)
np.random.seed(0)

# Check for ffmpeg installation
def check_ffmpeg():
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("ffmpeg is installed. Version info:")
            version_info = result.stdout.decode('utf-8').split('\n')[0]
            print(version_info)
            return True
        else:
            print("ffmpeg command found but returned error.")
            return False
    except Exception:
        print("ffmpeg not found. Animation saving may be limited.")
        return False

# Monkey patch the reward function to just return 0
def simple_reward(self):
    """Simplified reward function that just returns 0."""
    return 0.0

# Apply the monkey patches
MapEnv._get_reward = simple_reward

def test_map_env_with_random_actions(episodes=2, max_steps_per_episode=50, render=True, 
                                     save_animation=True, env_seed=42, episode_seeds=None):
    """
    Test the MapEnv with random actions to verify all components work together.
    Using a simplified reward function that just returns 0.
    
    Args:
        episodes: Number of episodes to run
        max_steps_per_episode: Maximum steps per episode
        render: Whether to render the environment during testing
        save_animation: Whether to save animations
        env_seed: Initial seed for environment creation
        episode_seeds: List of seeds for each episode (if None, will generate)
    """
    # Set up seeds - use clearly different values
    if episode_seeds is None:
        # Generate very different seeds for each episode
        episode_seeds = [env_seed + 1000 * (i+1) for i in range(episodes)]
    
    print(f"Using environment seed: {env_seed}")
    print(f"Episode seeds: {episode_seeds}")
        
    # Create directory for saving test artifacts
    output_dir = "test_results"
    if os.path.exists(output_dir):
        # Clean directory to avoid old files interfering
        print(f"Cleaning {output_dir} directory...")
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error cleaning {file_path}: {e}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for ffmpeg
    has_ffmpeg = check_ffmpeg()
    if not has_ffmpeg and save_animation:
        print("Note: MP4 animations may not work, but we'll try to save GIFs instead.")
    
    try:
        # Initialize environment with sequential observation space
        print(f"\n=== Creating environment with seed {env_seed} ===")
        env = MapEnv(
            location_name="Austin, Texas, USA",
            airspace_tag_list=[("amenity", "hospital"), ("aeroway", "aerodrome")],
            max_uavs=4,
            max_vertiports=6,
            max_episode_steps=max_steps_per_episode,
            seed=env_seed,  # Use the specified environment seed
            obs_space_str="seq",
            sorting_criteria="closest first",
            render_mode="human" if render else None
        )
        
        for episode in range(episodes):
            # Use a very different seed for each episode
            episode_seed = episode_seeds[episode]
            print(f"\n===== Starting Episode {episode+1}/{episodes} with seed {episode_seed} =====")
            
            try:
                # Reset environment with episode-specific seed
                print(f"Resetting environment with seed {episode_seed}...")
                obs, info = env.reset(seed=episode_seed)
                print(f"Initial observation keys: {obs.keys() if isinstance(obs, dict) else 'not a dict'}")
                
                steps = 0
                collision_detected = False
                goal_reached = False
                
                # Run episode
                for step in range(max_steps_per_episode):
                    # Sample random action
                    action = env.action_space.sample()
                    
                    try:
                        # Take step in environment
                        next_obs, reward, terminated, truncated, info = env.step(action)
                        steps += 1
                        
                        # Print occasional status
                        if step % 5 == 0:  # More frequent updates for shorter episodes
                            goal_dist = info.get('distance_to_goal', 'N/A')
                            if not isinstance(goal_dist, str):
                                goal_dist = f"{goal_dist:.2f}"
                            print(f"Step {step}: action = [{action[0]:.2f}, {action[1]:.2f}], "
                                  f"distance to goal = {goal_dist}")
                        
                        # Optionally render
                        if render:
                            try:
                                env.render()
                                time.sleep(0.05)  # Slightly longer pause for visibility
                            except Exception as e:
                                print(f"Render error: {e}")
                                traceback.print_exc()
                                render = False
                        
                        # Check termination conditions
                        if terminated:
                            print(f"Goal reached at step {step}!")
                            goal_reached = True
                            break
                            
                        if truncated:
                            if 'collision' in info:
                                print(f"Collision detected at step {step}!")
                                collision_detected = True
                            elif 'timeout' in info and info['timeout']:
                                print(f"Episode timeout after {step} steps!")
                            break
                    
                    except Exception as e:
                        print(f"Error during step {step}:")
                        traceback.print_exc()
                        print(f"Continuing to next episode...")
                        break
                
                # Episode summary
                print(f"\nEpisode {episode+1} summary:")
                print(f"Total steps: {steps}")
                print(f"Goal reached: {goal_reached}")
                print(f"Collision detected: {collision_detected}")
                
                # Create animation for this episode
                if save_animation and steps > 0:
                    try:
                        print("\nCreating animation...")
                        # Create animation with the full number of steps taken
                        ani = env.create_animation(steps)
                        if ani:
                            save_path = os.path.join(output_dir, f"episode_{episode+1}_seed_{episode_seed}_animation")
                            print(f"Saving animation to {save_path}")
                            env.save_animation(ani, save_path)
                            
                            # Check if animation was successfully saved
                            gif_path = f"{save_path}.gif"
                            mp4_path = f"{save_path}.mp4"
                            
                            if os.path.exists(gif_path):
                                print(f"GIF successfully saved at: {gif_path}")
                                file_size = os.path.getsize(gif_path) / (1024 * 1024)  # Size in MB
                                print(f"GIF file size: {file_size:.2f} MB")
                            else:
                                print(f"Warning: GIF file not found at {gif_path}")
                                
                            if os.path.exists(mp4_path):
                                print(f"MP4 successfully saved at: {mp4_path}")
                                file_size = os.path.getsize(mp4_path) / (1024 * 1024)  # Size in MB
                                print(f"MP4 file size: {file_size:.2f} MB")
                            else:
                                print(f"Warning: MP4 file not found at {mp4_path}")
                        else:
                            print("Failed to create animation - no animation object returned")
                    except Exception as e:
                        print(f"Animation error: {e}")
                        traceback.print_exc()
            
            except Exception as e:
                print(f"Error during episode {episode+1}:")
                traceback.print_exc()
                print(f"Continuing to next episode...")
                continue
        
        # Close environment
        env.close()
        print("\nEnvironment closed!")
        
        # Analyze the generated logs using the integrated loader
        print("\n=== Analyzing Generated Logs ===")
        loader = MapLoader(base_log_dir="logs")
        
        episodes = loader.list_episodes()
        if episodes:
            print(f"Found {len(episodes)} episode(s) in logs")
            
            # Analyze each episode
            for episode_dir in episodes:
                print(f"\nAnalyzing episode: {episode_dir}")
                
                # Print episode summary
                loader.print_episode_summary(episode_dir)
                
                # Get all agents
                non_learning_agents = loader.get_non_learning_agents(episode_dir)
                learning_agents = loader.get_learning_agents(episode_dir)
                
                print(f"\nNon-learning agents: {non_learning_agents}")
                print(f"Learning agents: {learning_agents}")
                
                # Print details for each agent
                for agent_id in non_learning_agents + learning_agents:
                    loader.print_agent_details(episode_dir, agent_id)
        else:
            print("No episodes found in logs!")
                
        print("\nTest completed!")
    
    except Exception as e:
        print("Fatal error during testing:")
        traceback.print_exc()


if __name__ == "__main__":
    print("Testing MapEnv with random actions and enhanced animation capabilities...")
    
    # Check if we can import necessary libraries
    try:
        import matplotlib.animation
        has_animation = True
    except ImportError:
        has_animation = False
        print("Warning: matplotlib.animation not available, animations will be disabled")
    
    # Define very different seeds to clearly demonstrate seed effect
    env_seed = 17  # Main environment seed
    episode_seeds = [70, 85]  # Completely different seeds for each episode
    
    # Increase to 500 steps to see more movement
    test_map_env_with_random_actions(
        episodes=2,
        max_steps_per_episode=500,
        render=True,
        save_animation=has_animation,
        env_seed=env_seed,
        episode_seeds=episode_seeds
    )
    
    print("Test script completed.")