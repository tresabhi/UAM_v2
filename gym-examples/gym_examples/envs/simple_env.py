from  uav_v2 import UAV_v2
from controller_static import StaticController  
from controller_non_coop import NonCoopController
from controller_non_coop_smooth import NonCoopControllerSmooth
from dynamics_point_mass import PointMassDynamics
from sensor_universal import UniversalSensor
from space import Space
import math
from shapely import Point
from matplotlib import pyplot as plt 
import gymnasium as gym 


class SimpleEnv(gym.Env):
    def __init__(self, ): #! add UAVs, vertiports, and other parameters that are needed for space.
        super().__init__()
        
        # env needs to initialzed with number of UAVs, vertiports, and some parameters that are needed for space.
        # The parameters will be used by methods from space to create UAVs, vertiports, assign start-end points, etc.
        
        self.space = Space(max_uavs=2, max_vertiports=2)
        universal_sensor = UniversalSensor(space=self.space)
        static_controller = StaticController(0,0)
        non_coop_smooth_controller = NonCoopControllerSmooth(10,2)
        non_coop_controller = NonCoopController(10,1)
        pm_dynamics = PointMassDynamics()
        universal_sensor = UniversalSensor(space=self.space)

    def step(self, action):
        # this method will accept action from model, and apply it to agent
        # this will update dynamics of agent, and return observation, reward, done, info
        # this will also update dynamics of other agents, and check for collision, and update info
        pass

    def reset(self):
        # if agent has collision, call reset.
        # if agent reaches goal, call reset.
        # if agent reaches max steps, call reset.
        
        # this method will recreate the environment and return initial observation,
        # where every other agent is at their start point, 
        # no collision has happend, and no agent has reached the goal. 
        pass

    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass    
    
    def get_reward(self):
        '''A sparse reward function that rewards the agent for reaching the goal.'''
        pass



