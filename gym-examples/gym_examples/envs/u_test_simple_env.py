from simple_env import SimpleEnv
from SARSA_logger import SARSALogger
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# CREATE the environment and logger
env = SimpleEnv(12, 14, 7, 42, 'seq', 'closest first')
logger = SARSALogger()
print("Environment and logger created successfully")

# RESET the environment and get initial observation
current_obs, _ = env.reset()
print("\nInitial Observation:")
print(current_obs)

# Get initial action
current_action = env.action_space.sample()

# Run simulation for multiple steps
num_steps = 20000
step_delay = 0.1

print("\nStarting simulation...")
print("Running for", num_steps, "steps with random actions")

try:
    step = 0
    while step < num_steps:
        # Render the current state
        env.render()
        
        # Take a step in the environment
        print(f'\n current time step: {step}')
        next_obs, reward, done, info = env.step(current_action)
        
        # Sample next action (for SARSA)
        next_action = env.action_space.sample()
        
        # Log the SARSA transition
        logger.log_transition(
            state=current_obs,
            action=current_action,
            reward=reward,
            next_state=next_obs,
            next_action=next_action,
            info=info
        )
        
        # Update current state and action
        current_obs = next_obs
        current_action = next_action
        
        # Add a small delay for visualization
        time.sleep(step_delay)
        
        # Print step information every 50 steps
        if step % 50 == 0:
            print(f"\nStep {step}:")
            print(f"Action taken: {current_action}")
            print(f"Reward: {reward}")
        
        # If the episode is done, log it and reset
        if done:
            print("\nEpisode finished due to:", info)
            logger.end_episode(
                success='reached_goal' in info and info['reached_goal'],
                truncated='collision' in info or 'timeout' in info
            )
            print("Resetting environment...")
            current_obs, _ = env.reset()
            current_action = env.action_space.sample()
        
        step += 1

except KeyboardInterrupt:
    print("\nSimulation interrupted by user")
finally:
    # Clean up
    logger.close()
    print("\nSimulation complete")
    env.close()
    plt.close('all')