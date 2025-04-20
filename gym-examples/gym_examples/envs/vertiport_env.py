# veriport_env.py

from vertiport import Vertiport
from gymnasium import spaces
import gymnasium



class VertiportEnv(gymnasium.Env):

    def __init__(self, ):
        super().__init__()

        self.uav_list = None # list of UAVs in the env
        self.vertiport_list = None # How do we pick the vertiport list for the environement to start its RL process?


    
    def reset(self,):
        pass


    def step(self,):
        pass



    def render(self,):
        pass


    def close(self,):
        pass





