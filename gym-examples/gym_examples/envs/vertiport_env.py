# veriport_env.py

from vertiport import Vertiport
from gymnasium import spaces
import gymnasium
import networkx as nx 


class VertiportEnv(gymnasium.Env):

    def __init__(self, ):
        super().__init__()

        self.uav_list = None # list of UAVs in the env
        self.vertiport_list = None # How do we pick the vertiport list for the environement to start its RL process?


    def _create_vertiport_graph():
        # look into how google_deepmind's implementation 
        pass

    
    def reset(self,):
        pass


    def _get_reward():
        # Nameer and Aadit's initial elements for reward:
        # 1. reduce congestion
        # 2. reduce travel time 
        # 3. reduce collision/NMAC incidence 
        # 4. reduce overall sound  
        pass

    def step(self,):
        pass



    def render(self,):
        pass


    def close(self,):
        pass





