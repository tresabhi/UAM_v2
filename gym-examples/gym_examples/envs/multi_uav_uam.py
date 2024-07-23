
import functools
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import time
from geopandas import GeoSeries
import geopandas as gpd
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv

from airspace import Airspace
from airtrafficcontroller import ATC
from autonomous_uav import Autonomous_UAV
from vertiport import Vertiport



class Uam_Uav_Env_PZ(ParallelEnv):
    metadata = {
        "name": "multi_uav_uam_v0",
    }

    def __init__(self, location_name, num_vertiports, num_auto_uav, sleep_time = 0.05, render_mode=None):
        #Environment attributes 
        self.current_time_step = 0
        self.num_vertiports = num_vertiports
        self.num_auto_uav = num_auto_uav
        self.sleep_time = sleep_time
        self.airspace = Airspace(location_name)
        self.atc = ATC(airspace=self.airspace)

        #Vertiport Initialization 
        self.atc.create_n_random_vertiports(self.num_vertiports)

        #Auto UAV initialization 
        #make a list of start_vertiport 
        #make a list of end vertiport 
        #make a list of AutoUAV
        #make an attribute -> self.auto_uavs = {auto_uav.id:auto_uav for auto_uav in list_AUTO_UAV}

        #Petting Zoo API attributes 
        self.possible_agents = list(self.auto_uavs.keys()) #.keys() because auto_uavs will need to be a dictionary 
        self.observation_spaces = {}
        self.action_spaces = {}

    def reset(self, seed=None, options=None):
        pass

    def step(self, actions):
        pass

    def render(self):
        pass
    
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, auto_uav:Autonomous_UAV):
        return spaces.Dict(
        {
            # Agent ID as integer, using smaller space for IDs
            "agent_id": spaces.Box(
                low=0, high=np.iinfo(np.int32).max, shape=(1,), dtype=np.int32
            ),

            # Agent speed
            "agent_speed": spaces.Box( 
                low=0,  # speed should be non-negative
                high=auto_uav.max_speed,
                shape=(1,),
                dtype=np.float64,
            ),

            # Agent deviation, corrected to -180 to 180
            "agent_deviation": spaces.Box(
                low=-180, high=180, shape=(1,), dtype=np.float64
            ),

            # Intruder detection
            "intruder_detected": spaces.Discrete(2),  # 0 for no intruder, 1 for intruder detected

            # Intruder ID, using smaller space for IDs
            "intruder_id": spaces.Box(
                low=0, high=np.iinfo(np.int32).max, shape=(1,), dtype=np.int32
            ),

            # Distance to intruder
            "distance_to_intruder": spaces.Box(
                low=0,
                high=auto_uav.detection_radius,
                shape=(1,),
                dtype=np.float64,
            ),

            # Relative heading of intruder, corrected to -180 to 180
            "relative_heading_intruder": spaces.Box(
                low=-180, high=180, shape=(1,), dtype=np.float64
            ),

            # Intruder's heading
            "intruder_current_heading": spaces.Box(
                low=-180, high=180, shape=(1,), dtype=np.float64
            ),
        }
    )
    
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(
                        low=-1, 
                        high=1, 
                        shape=(2,), 
                        dtype=np.float64
    )
    