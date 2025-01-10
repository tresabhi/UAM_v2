from uav_v2 import UAV_v2
from auto_uav_v2 import Auto_UAV_v2
from controller_static import StaticController  
from controller_non_coop import NonCoopController
from controller_non_coop_smooth import NonCoopControllerSmooth
from dynamics_point_mass import PointMassDynamics
from sensor_universal import UniversalSensor
from space import Space
import math
import numpy as np
from shapely import Point
from matplotlib import pyplot as plt 
import gymnasium as gym 
from gymnasium.spaces import Discrete, Box, Dict
from gymnasium import spaces


class SimpleEnv(gym.Env):
    def __init__(self, max_uavs=12, max_vertiports=14, seed = 42): #! add UAVs, vertiports, and other parameters that are needed for space.
        super().__init__()
        
        # env needs to initialzed with number of UAVs, vertiports, and some parameters that are needed for space.
        # The parameters will be used by methods from space to create UAVs, vertiports, assign start-end points, etc.
        self.max_uavs = max_uavs
        self.max_vertiports = max_vertiports
        self._seed = seed


        self.observation_space = Dict({ #! I think I should add end_vertiport co-ord ??
                                        'no_other_agents': Discrete(9),
                                        'dist_goal': Box(low=0, high=250, shape=(), dtype=np.float32),
                                        'heading_ego_frame': Box(low=-180, high=180, shape=(), dtype=np.float32),
                                        'pref_speed': Box(low=0, high=25, shape=(), dtype=np.float32),
                                        'radius': Box(low=0, high=20, shape=(), dtype=np.float32),
                                        'other_agent_state': Box(
                                                                    low=np.full((8, 8), -np.inf),  # Use -inf as the lower bound for unspecified dimensions
                                                                    high=np.full((8, 8), np.inf),  # Use inf as the upper bound for unspecified dimensions
                                                                    shape=(8, 8),  # Array size 8x8
                                                                    dtype=np.float32  # Ensure consistent data type
                                                                )
                                    })
        self.action_space = spaces.Box(low=np.array([-1, -math.pi]),  # acceleration, heading_change
                                       high=np.array([1, math.pi]), 
                                       shape=(2,))
        
        

    def step(self, action):
        # this method will accept action from model, and apply it to agent
        # this will update dynamics of agent, and return observation, reward, done, info
        # this will also update dynamics of other agents, and check for collision, and update info
        for uav in self.space.get_uav_list():
            if isinstance(uav, UAV_v2):
                # get_state only returns UAVs personal state
                uav_state = uav.get_state() #! this is not required 

                # NMAC
                is_nmac, nmac_list = uav.sensor.get_nmac(uav)
                if is_nmac:
                    print('--- NMAC ---')
                    print(f'NMAC detected:{is_nmac}, and NMAC with {nmac_list}\n')
                
                # Collision
                is_collision, collision_uav_ids = uav.sensor.get_collision(uav)
                if is_collision:
                    print('---COLLISION---')
                    print(f'Collision detected:{is_collision}, and collision with {collision_uav_ids}\n')
                    self.space.remove_uavs_by_id(collision_uav_ids)
                    if len(self.space.uav_list) == 0:
                        print('NO more uavs in space')
                        end_sim = True
                        break
                

                print(f'Observation\n{uav_state}\n')
                # place inside step of gym.env
                
                # get_observation returns a list of dicts - first one is self state,
                # and the rest are other UAVs state. 
                observation = uav.get_obs()
                uav_mission_complete_status = uav.get_mission_status()
                uav.set_mission_complete_status(uav_mission_complete_status)
                uav_action = uav.get_action(observation=observation)
                uav.dynamics.update(uav, uav_action)

            else:
                # the structure of the code is same as above, but the agent is different.
                # the agent will go throguh the same process of performing -
                # get_nmac, get_collision, update_dynamics(action), get_obs, get_mission_status.
                self.agent.dynamic.update(self.agent, action)
            
            obs = self._get_obs() # once the agent has updated its dynamics, get the observation
            reward = self._get_reward() # 
            done = None # this signal comes from get_mission_status
            info = self._get_info()

            return obs, reward, done, info 


    def _get_reward(self):
        pass

    def _get_obs(self):
        pass 

    def _get_info(self):
        pass      
                
        

    def reset(self):
        # if agent has collision, call reset.
        # if agent reaches goal, call reset.
        # if agent reaches max steps, call reset.
        self.space = Space(max_uavs=self.max_uavs, max_vertiports=self.max_vertiports, seed=self._seed)
        self.universal_sensor = UniversalSensor(space=self.space)
        self.static_controller = StaticController(0,0)
        self.non_coop_smooth_controller = NonCoopControllerSmooth(10,2)
        self.non_coop_controller = NonCoopController(10,1)
        self.pm_dynamics = PointMassDynamics()
        self.universal_sensor = UniversalSensor(space=self.space)


        # --- Vertiport construction ---  
        self.space.create_circular_pattern_vertiports(8,300)
        # self.space.create_random_pattern_vertiports(8,300)
        
        # --- UAV construction ---
        self.space.create_uavs(4, UAV_v2, has_agent=True, controller=self.non_coop_smooth_controller, dynamics=self.pm_dynamics, sensor=self.universal_sensor, radius=5, nmac_radius=20)

        # --- UAV start-end assignment ---
        self.space.assign_vertiports('opposite')


        # --- create Agent (auto UAV) ---
        self.agent  = Auto_UAV_v2(dynamics=self.pm_dynamics, sensor=self.universal_sensor, radius=5, nmac_radius=20)

        # ---Agent (Auto UAV) start-end assignment ---
        self.space.set_uav(self.agent)
        self.space.assign_vertiport_agent(self.agent)

        obs = self._get_obs() #self.agent.get_obs()
        info = self._get_info() #self.agent.get_info()

        return obs, info 




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
    
    



