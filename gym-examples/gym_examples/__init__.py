from gymnasium.envs.registration import register

register(
     id="gym_examples/UAM-v0",
     entry_point="gym_examples.envs:Uam_Uav_Env",
     max_episode_steps=10_000,
)