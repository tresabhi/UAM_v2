from simple_env import SimpleEnv



env = SimpleEnv(12,14,7,42,'seq', 'closest first')

obs, _ = env.reset()

print(obs)

action_sample = env.action_space.sample()
print(action_sample)

env.step(action_sample)