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
from utils_data_transform import transform_sensor_data


class SimpleEnv(gym.Env):


    max_number_other_agents_observed = 7 #! at max 7 UAV for seq based model, else for graph based model - any number of UAV within detection radius  

    # Local variables 
    obs_space_seq = Dict({ #! I think I should add end_vertiport co-ord ??
                                        'no_other_agents': Discrete(max_number_other_agents_observed),
                                        'dist_goal': Box(low=0, high=250, shape=(), dtype=np.float32),
                                        'heading_ego_frame': Box(low=-180, high=180, shape=(), dtype=np.float32),
                                        'current_speed': Box(low=0, high=25, shape=(), dtype=np.float32),
                                        'radius': Box(low=0, high=20, shape=(), dtype=np.float32), # UAV size
                                        'other_agent_state': Box(   # p_parall, p_orth, v_parall, v_orth, other_agent_radius, combined_radius, dist_2_other
                                                                    low=np.full((max_number_other_agents_observed, 7), -np.inf),  # Use -inf as the lower bound for unspecified dimensions
                                                                    high=np.full((max_number_other_agents_observed, 7), np.inf),  # Use inf as the upper bound for unspecified dimensions
                                                                    shape=(max_number_other_agents_observed, 7),  # Array size (other_observed_uav, 7)
                                                                    dtype=np.float32  # Ensure consistent data type
                                                                )
                                    })


    obs_space_graph = spaces.Dict({
                                    'num_other_agents': spaces.Box(low=0, high=100, shape=(), dtype=np.int64),
                                    'agent_dist_to_goal': spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
                                    'agent_end_point': spaces.Box(low=np.array([-np.inf, -np.inf]), high=np.array([np.inf, np.inf]), shape=(2,), dtype=np.float32),
                                    'agent_current_position': spaces.Box(low=np.array([-np.inf, -np.inf]), high=np.array([np.inf, np.inf]), shape=(2,), dtype=np.float32),
                                    'graph_feat_matrix': spaces.Box(
                                        low=np.full((max_number_other_agents_observed + 1, 5), -np.inf),  # +1 for the host UAV
                                        high=np.full((max_number_other_agents_observed + 1, 5), np.inf),
                                        shape=(max_number_other_agents_observed + 1, 5),
                                        dtype=np.float32
                                    ),
                                    'edge_index': spaces.Box(
                                        low=0,
                                        high=max_number_other_agents_observed,
                                        shape=(2, max_number_other_agents_observed),
                                        dtype=np.int64
                                    ),
                                    'edge_attr': spaces.Box(
                                        low=0,
                                        high=np.inf,
                                        shape=(max_number_other_agents_observed, 1),  # Assuming 1 feature per edge (e.g., distance)
                                        dtype=np.float32
                                    ),
                                    'mask': spaces.Box(
                                        low=0,
                                        high=1,
                                        shape=(max_number_other_agents_observed + 1,),  # +1 for the host UAV
                                        dtype=np.float32
                                    )
                                })


    def __init__(self, max_uavs=12, max_vertiports=14,
                 max_number_other_agents_observed = 7, 
                 seed = 42, 
                 obs_space_str= None, 
                 sorting_criteria=None): #! add UAVs, vertiports, and other parameters that are needed for space.
        super().__init__()

        # env obs chech 
        if obs_space_str=='seq' and sorting_criteria == None:
            raise RuntimeError('env.__init__ needs both obs_space_str and sorting_criteria for sequential observation space')
        
        # env needs to initialzed with number of UAVs, vertiports, and some parameters that are needed for space.
        # The parameters will be used by methods from space to create UAVs, vertiports, assign start-end points, etc.
        self.max_uavs = max_uavs
        self.max_vertiports = max_vertiports
        self.max_number_other_agents_observed = max_number_other_agents_observed
        self._seed = seed
        self.obs_space_str = obs_space_str
        self.sorting_criteria = sorting_criteria

        if self.obs_space_str == 'seq':
            self.observation_space = SimpleEnv.obs_space_seq
        elif self.obs_space_str == 'graph':
            self.observation_space = SimpleEnv.obs_space_graph
        else:
            raise RuntimeError('Choose correct format of obs space and provide correct string to init')
        
        self.action_space = spaces.Box(low=np.array([-1, -math.pi]),  # acceleration, heading_change
                                       high=np.array([1, math.pi]), 
                                       shape=(2,))
        
        

    def step(self, action):
        # this method will accept action from model, and apply it to agent
        # this will update dynamics of agent, and return observation, reward, done, info
        # this will also update dynamics of other agents, and check for collision, and update info
        for uav in self.space.get_uav_list():
            if isinstance(uav, UAV_v2): #! checking using class type, is this a good way to check this?
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
                

                print(f'Observation\n{uav_state}\n') #! this will need to be updated if uav_state is deleted
                # place inside step of gym.env
                
                # get_observation returns a list of dicts - first one is self state,
                # and the rest are other UAVs state. 
                observation = uav.get_obs() #! this method will accept an argument - obs_space_str, this observation should have the same format as the agent/env - this will be used for supervised training 
                uav_mission_complete_status = uav.get_mission_status()
                uav.set_mission_complete_status(uav_mission_complete_status)
                #! need to update get_action method, so that, it transforms the observations into gym_space observation for supervised training
                uav_action = uav.get_action(observation=observation) 
                #! need a way to store observation and action pair for supervised training  
                uav.dynamics.update(uav, uav_action)

            else:
                # the structure of the code is same as above, but the agent is different.
                # the agent will go throguh the same process of performing -
                # get_nmac, get_collision, update_dynamics(action), get_obs, get_mission_status.
                self.agent.dynamic.update(self.agent, action)
            
            #! the obs, reward, info needs to be of the correct format, based on env - seq or graph
            #! _get_obs() is for agent, which will accept obs_str and pass it to all relevant methods of agent that will create its obs.
            obs = self._get_obs() # once the agent has updated its dynamics, get the observation
            reward = self._get_reward() # 
            done = None # this signal comes from get_mission_status
            info = self._get_info()

            return obs, reward, done, info 


    def _get_reward(self):
        pass
    
    #! _get_obs() is for agent, which will accept obs_str and pass it to all relevant methods of agent that will create its obs.
    def _get_obs(self) -> spaces.Dict:
        '''Returns observation of the agent in a specific format'''
        if self.obs_space_str == 'seq':
            # get Auto-UAV observation data
            raw_obs = self.agent.get_obs()
            # pass observation to transform_data
            transformed_data = transform_sensor_data(raw_obs, self.max_number_other_agents_observed,'seq', self.sorting_criteria) 
            # return transformed obs_data
            return transformed_data
        elif self.obs_space_str == 'graph':
            # get Auto-UAV observation data
            raw_obs = self.agent.get_obs()
            # pass observation to transform_data
            transformed_data = transform_sensor_data(raw_obs, self.max_number_other_agents_observed, 'graph')
            # return transformed obs_data
            return transformed_data
        else:
            raise RuntimeError('_get_obs \n incorrect self.obs_space_str, check __init__ and self.obs_space_str')

         

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

        #! _get_obs() is for agent, which will accept obs_str and pass it to all relevant methods of agent that will create its obs.
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
    
    



