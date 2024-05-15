import numpy as np 
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

from wrappers import airspace, airtrafficcontroller

uam_airspace = airspace.Airspace('Austin, Texas, USA')
uam_atc = airtrafficcontroller.ATC(airspace=uam_airspace)

#! Warning
'''
Remember this environment will only be used for training,
Therefore there will only be one auto_uav which is built into the uam_env 
'''
#! End 


class Uam_Uav_Env(gym.Env):
    metadata = {"render_mode":["human", "rgb_array"], "render_fps":4}

    def __init__(self, location_name, num_vertiport, num_reg_uav,render_mode=None):
        #airspace,airtrafficcontroller,uav,autonomous_uav,vertiport,
        pass 

    def _get_obs(self,):
        pass 

    def _get_info(self,):
        pass 

    def reset(self,):
        pass 

    def step(self,action):
        pass

    def render(self,):
        pass

    def _render_frame(self,):
        pass

    def close(self,):
        pass
    