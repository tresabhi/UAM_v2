import numpy as np 
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

from wrappers import airspace, airtrafficcontroller

uam_airspace = airspace.Airspace('Austin, Texas, USA')
uam_atc = airtrafficcontroller.ATC(airspace=uam_airspace)



class Uam_Uav_Env(gym.Env):
    metadata = {"render_mode":["human", "rgb_array"], "render_fps":4}

    def __init__(self, airspace,airtrafficcontroller,uav,autonomous_uav,vertiport,render_mode=None):
        pass 

    def _get_obs(self,):
        pass 

    def _get_info(self,):
        pass 

    def reset(self,):
        pass 

    def step(self,action):
        pass
    