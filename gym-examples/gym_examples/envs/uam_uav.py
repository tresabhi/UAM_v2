import numpy as np 
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

from assets import airspace, airtrafficcontroller

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
        '''
        This method will collect observations for auto_uav only. 
        '''
        pass 

    def _get_info(self,):
        pass 

    def reset(self,):
        '''
        This method will reset the environment. NOT sure how this will be done, but it will be done.
        '''
        pass 

    def step(self,action):
        '''
        This method is used to step the environment, it will step the environment by one timestep.
        
        The action argument - will be passed to auto_uav's step method
        
        Regular UAVs will step without action. so I will need to modify regular uav in such a way that they will step without action. 
        This tells me that regular uav will need to have collision avoidance built into the UAV module, such that they can step without action. 
         
        '''
        return obs, reward, done, info

    def render(self,):
        pass

    def _render_frame(self,):
        pass

    def close(self,):
        pass
    