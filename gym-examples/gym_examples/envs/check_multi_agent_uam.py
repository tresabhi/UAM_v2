from multi_uav_uam import Uam_Uav_Env_PZ

from pettingzoo.test import parallel_api_test


env = Uam_Uav_Env_PZ('Austin, Texas, USA', 8, 5, sleep_time=0.01)
parallel_api_test(env, num_cycles=100)
