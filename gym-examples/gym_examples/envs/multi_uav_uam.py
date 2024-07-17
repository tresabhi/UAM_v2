
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

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]