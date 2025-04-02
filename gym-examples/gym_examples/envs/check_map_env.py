from map_env_revised import MapEnv

env = MapEnv(3,5, obs_space_str='UAM_UAV')
env.reset()

print(env.observation_space.shape)
print(env.action_space.shape)

print(env.observation_space.sample())
print(env.action_space.sample())