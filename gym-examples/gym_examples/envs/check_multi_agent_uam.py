from multi_uam_uav import UamUavEnvPZ

from pettingzoo.test import parallel_api_test


env = UamUavEnvPZ("Austin, Texas, USA", 8, 5, sleep_time=0.01)
parallel_api_test(env, num_cycles=100)
