# Test observation and action spaces
from uam_uav import UamUavEnv
import numpy as np
import matplotlib.pyplot as plt

def test_spaces():
    # Create environment
    env = UamUavEnv(
        "Austin, Texas, USA",
        8,
        5,
        sleep_time=0.01,
        airspace_tag_list=[("building", "hospital"), ("aeroway", "aerodrome")],
    )
    
    #! Error in observation space sampling, observation shape registers as None
    print("\n=== Testing Observation Space ===")
    print(f"Observation space shape: {env.observation_space.shape}")
    # sample_obs = env.observation_space.sample()
    # print("\nSample observation:")
    # for key, value in sample_obs.items():
    #     print(f"{key}: {value} (shape: {value.shape if hasattr(value, 'shape') else 'scalar'})")
    
    print("\n=== Testing Action Space ===")
    print(f"Action space shape: {env.action_space.shape}")
    action = env.action_space.sample()
    print(f"Sample action: {action}")
    
    # Test a few environment steps
    print("\n=== Testing Environment Steps ===")
    fig, ax = env.render_init()
    obs, _ = env.reset()
    nsteps = 0
    print("\nInitial observation ranges:")
    for key, value in obs.items():
        if hasattr(value, 'item'):
            print(f"{key}: {value.item()}")
        else:
            print(f"{key}: {value}")
    
    # Test multiple steps
    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render(fig,ax)

        if (i % 50 == 0):
            print(f"\nStep {i+1}:")
            print(f"Action taken: {action}")
            print(f"Reward: {reward}")
            print(f"Terminated: {terminated}")
            print(f"Truncated: {truncated}")
            print(f"Distance to goal: {info['distance_to_end_vertiport']:.2f}")
            for key, value in obs.items():
                if hasattr(value, 'item'):
                    print(f"{key}: {value.item()}")
                else:
                    print(f"{key}: {value}")

        if terminated:
            print(f"Episode ended in {i} steps, success!")
            nsteps = i
            break
        elif truncated:
            print("Episode truncated, resetting...")
            obs, _ = env.reset()

    if len(env.df) > 0:
        ani = env.create_animation(nsteps + 1)
        if ani:
            env.save_animation(ani, "test_episode")

    env.close()
    plt.close()

if __name__ == "__main__":
    test_spaces()