from simple_env import SimpleEnv



env = SimpleEnv(12,14,7,42,'seq', 'closest first')

obs, _ = env.reset()

print(obs)