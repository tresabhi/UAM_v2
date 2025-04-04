import math
import random
import numpy as np
import time
import pandas as pd
import geopandas as gpd

from uav_v2 import UAV_v2
from auto_uav_v2 import Auto_UAV_v2
from controller_static import StaticController
from controller_non_coop import NonCoopController
from controller_non_coop_smooth import NonCoopControllerSmooth
from dynamics_point_mass import PointMassDynamics
from map_sensor import MapSensor
from atc import ATC
from airspace import Airspace
from utils_data_transform import transform_sensor_data, choose_obs_space_constructor, transform_for_uam
from mapped_env_util import *

from shapely import Point
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict
from gymnasium import spaces
# from UAV_logger import NonLearningLogger
from map_logger import MapLogger
from map_renderer import MapRenderer

class MapEnv(gym.Env):
    """
    Urban Air Mobility environment for flying multiple UAVs with an autonomous agent
    in a real-world map with restricted airspace.
    """


    def __init__(
        self,
        number_of_uav = None, # number of UAV for this simulation 
        number_of_vertiport=None, # number of vertiorts for this simulation 
        location_name="Austin, Texas, USA", # name of city 
        airspace_tag_list=[("amenity", "hospital"), ("aeroway", "aerodrome")], # 
        max_episode_steps=1000, # number of gym steps per episode 
        number_of_other_agents_for_model=7, # max number of other agents that learning agent tracks for LSTM-A2C model 
        sleep_time=0.005,
        seed=70,
        obs_space_str=None, #! this will be used to choose the constructor 
        sorting_criteria=None, # this needs to be defined for LSTM-A2C model
        render_mode=None,
        max_uavs=8, # this is maximum number of UAVs allowed in env
        max_vertiports=12, # this is maximum number of vertiports allowed in env
    ):  
        super().__init__()
        
        # Environment configuration
        self.number_of_uav = number_of_uav
        self.number_of_vertiport = number_of_vertiport
        self.location_name = location_name
        self.airspace_tag_list = airspace_tag_list
        self.max_uavs = max_uavs
        self.max_vertiports = max_vertiports
        self.max_number_other_agents_observed = number_of_other_agents_for_model
        self.max_episode_steps = max_episode_steps
        self.sleep_time = sleep_time
        self._seed = seed
        self.obs_space_str = obs_space_str
        self.sorting_criteria = sorting_criteria
        self.render_mode = render_mode
        self.current_time_step = 0
        self.render_history = []  # Store positions for trajectory visualization

        # Initialize logger for non-learning agents
        self.logger = MapLogger(base_log_dir="logs")
        # Initialize renderer
        self.renderer = MapRenderer(self, render_mode, sleep_time)

        #FIX: cannot pass self.agent here, NOW WHAT
        self.observation_space = choose_obs_space_constructor(
            self.obs_space_str, 
            self.max_number_other_agents_observed
        )

        # Check observation space configuration
        if self.obs_space_str == "LSTM-A2C" and self.sorting_criteria is None:
            raise RuntimeError(
                "env.__init__ needs both obs_space_str and sorting_criteria "
                "for LSTM_A2C model ie sequential observation space"
            )
        
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),  # [acceleration, heading_change]
            high=np.array([1, 1]),
            shape=(2,),
            dtype=np.float64,
        )

    def step(self, action):
        #! why is the UAV detection not used, we only use nmac and collsion 
        #FUTURE: 
        # there needs to be two lists, one for learning_agents, and another for non-learning agents
        # if there is a collision we remove items from respective lists
        # collect information from the removed items
        # and continue until episode ends or agent list is empty(empty agent list means all agents had collision) 
        """
        Execute one time step within the environment.
        
        Args:
            action: Action provided by the learning agent [acceleration, heading_change]
            
        Returns:
            obs: Observation of the learning agent's state
            reward: Reward signal
            terminated: Whether the episode has ended due to goal reached
            truncated: Whether the episode has ended due to collision or timeout
            info: Additional information
        """
        # Update learning agent with provided action
        self.agent.dynamics.update(self.agent, action)
        
        # Save animation data for all UAVs
        self.renderer.add_data(self.agent)
        
        # Check collision with static objects (restricted airspace)
        static_collision, static_collision_info = self.agent.sensor.get_ra_collision(self.agent)
        if static_collision:
            print("--- Restricted Airspace COLLISION ---")
            print(f"Collision with restricted airspace: {static_collision_info}")
            self.logger.record_collision([self.agent.id], 'static')

            # Clean up trajectory data for collided agent
            if hasattr(self, 'trajectory_by_id') and self.agent.id in self.trajectory_by_id:
                del self.trajectory_by_id[self.agent.id]
            #      observation,    reward, truncated, terminated, info
            return self._get_obs(), -100,   False,     True,       {"collision": True, "type": "static"}
            
        # Check NMAC and collision for learning agent with other UAVs
        is_nmac, nmac_list = self.agent.sensor.get_nmac(self.agent)
        if is_nmac:
            print("--- NMAC ---")
            print(f"NMAC detected: {is_nmac}, and NMAC with {nmac_list}\n")
        
        is_collision, collision_uav_ids = self.agent.sensor.get_uav_collision(self.agent)
        if is_collision:
            print("--- COLLISION ---")
            print(f"Collision detected: {is_collision}, and collision with {collision_uav_ids}\n")
            self.logger.record_collision(collision_uav_ids, 'dynamic')

            # Clean up trajectory data for collided agents
            if hasattr(self, 'trajectory_by_id'):
                for uav_id in collision_uav_ids:
                    if uav_id in self.trajectory_by_id:
                        del self.trajectory_by_id[uav_id]

            self.atc.remove_uavs_by_id(collision_uav_ids)
            if self.atc.get_uav_list() == 0:
                print("No more UAVs in space")
                #       obs,            reward, terminated, truncated, info
                return self._get_obs(), -100,   False,      True,      {"collision": True, "type": "dynamic"}
        
        
        #### UAV (non-learning) ####

        
        # Now update all non-learning UAVs
        for uav in self.atc.get_uav_list():
            if not isinstance(uav, Auto_UAV_v2):  # Only process non-learning UAVs
                # Save animation data
                self.renderer.add_data(uav)
                
                # Get non-learning UAV's observations and check for collisions
                observation = uav.get_obs()

                # Check static collisions for non-learning UAVs
                static_collision, static_info = uav.sensor.get_ra_collision(uav)
                if static_collision:
                    print(f"--- UAV {uav.id} Restricted Airspace COLLISION ---")
                    self.logger.record_collision([uav.id], 'static')

                    # Clean up trajectory for this UAV
                    if hasattr(self, 'trajectory_by_id') and uav.id in self.trajectory_by_id:
                        del self.trajectory_by_id[uav.id]

                    self.atc.remove_uavs_by_id([uav.id])
                    continue
                
                # Check NMAC and dynamic collisions
                is_nmac, nmac_list = uav.sensor.get_nmac(uav)
                if is_nmac:
                    print(f"--- UAV {uav.id} NMAC ---")
                
                is_collision, collision_uav_ids = uav.sensor.get_uav_collision(uav)
                if is_collision:
                    print(f"--- UAV {uav.id} COLLISION ---")
                    self.logger.record_collision(collision_uav_ids, 'dynamic')

                    # Clean up trajectory for collided UAVs
                    if hasattr(self, 'trajectory_by_id'):
                        for coll_id in collision_uav_ids:
                            if coll_id in self.trajectory_by_id:
                                del self.trajectory_by_id[coll_id]

                    self.atc.remove_uavs_by_id(collision_uav_ids)
                    continue
                
                # Update non-learning UAV's state
                uav_mission_complete_status = uav.get_mission_status()
                uav.set_mission_complete_status(uav_mission_complete_status)

                # If UAV completed its mission, log it and clean up its trajectory
                if uav_mission_complete_status:
                    self.logger.mark_agent_complete(uav.id)
                    if hasattr(self, 'trajectory_by_id') and uav.id in self.trajectory_by_id:
                        del self.trajectory_by_id[uav.id]
                    self.atc.remove_uavs_by_id([uav.id])
                    continue
                
                # Get and apply non-learning UAV's action based on its controller
                uav_action = uav.get_action(observation=observation)
                uav.dynamics.update(uav, uav_action)

                # Log the state-action pair for this non-learning UAV
                self.logger.log_non_learning_step(uav.id, observation, uav_action)
        
        # Get learning agent's observation, reward, and status
        obs = self._get_obs()
        reward = self._get_reward()

        # Log the full transition for the learning agent
        if hasattr(self, 'previous_obs') and hasattr(self, 'previous_action'):
            self.logger.log_learning_transition(
                self.agent.id,
                self.previous_obs,
                self.previous_action,
                reward,
                obs,  # Current obs becomes next_state
                action  # Current action becomes next_action
            )
        else:
            # First step in episode, can't log transition yet
            pass
            
        # Store current state and action for next transition
        self.previous_obs = obs
        self.previous_action = action
        
        # Get mission status and set termination and truncation flags
        agent_mission_complete = self.agent.get_mission_status()
        terminated = agent_mission_complete
        timeout = self.current_time_step >= self.max_episode_steps
        truncated = timeout

        # If agent completed mission, log ut and clean up its trajectory
        if agent_mission_complete:
            self.logger.mark_agent_complete(self.agent.id)
            if hasattr(self, 'trajectory_by_id') and self.agent.id in self.trajectory_by_id:
                del self.trajectory_by_id[self.agent.id]
        
        # Prepare info dictionary
        info = {
            'distance_to_goal': self.agent.current_position.distance(self.agent.end),
            'current_step': self.current_time_step,
            'timeout': timeout
        }
        
        # Add static object detection info
        ra_data = self.agent.sensor.get_ra_detection(self.agent)
        if len(ra_data):
            info['static_detection'] = True
            info['distance_to_restricted'] = ra_data['distance']
        
        # Increment time step
        self.current_time_step += 1
        
        return obs, reward, terminated, truncated, info

    def _get_reward(self):
        #FIX: 
        # Depending on the type of obs_constructor used,
        # this method will need to choose the correct reward function
        # reward functions should be defined in mapped_env_utils 

        # """Calculate the reward for the learning agent."""
        # reward = 0.0
        
        # # Get current state
        # current_speed = self.agent.current_speed
        # current_distance = self.agent.current_position.distance(self.agent.end)
        
        # # Progress reward
        # if hasattr(self, 'previous_distance') and self.previous_distance is not None:
        #     progress = self.previous_distance - current_distance
        #     # Exponential scaling for distance factor to emphasize final approach
        #     distance_factor = np.exp(-current_distance / 5000)
        #     reward += progress * 15.0 * (1.0 + distance_factor)
        # self.previous_distance = current_distance
        
        # # Heading efficiency reward
        # ref_heading = math.atan2(
        #     self.agent.end.y - self.agent.current_position.y,
        #     self.agent.end.x - self.agent.current_position.x
        # )
        # heading_diff = ((math.degrees(ref_heading) - math.degrees(self.agent.current_heading) + 180) % 360) - 180
        # heading_efficiency = np.cos(np.deg2rad(heading_diff))
        
        # # Static collision avoidance rewards
        # static_detection, static_info = self.agent.sensor.get_static_detection(self.agent)
        # if static_detection:
        #     # Strong penalty for being near restricted airspace
        #     distance_to_restricted = static_info.get('distance', float('inf'))
        #     distance_factor = 1.0 - min(1.0, distance_to_restricted / self.agent.detection_radius)
        #     reward -= distance_factor * 50.0
            
        #     # Reduce heading importance during avoidance
        #     if abs(heading_diff) < 30.0:
        #         reward += heading_efficiency * 1.0  # Reduced reward during avoidance
        # else:
        #     # Normal heading rewards when no threats
        #     if abs(heading_diff) < 5.0:
        #         reward += 5.0
        #     elif abs(heading_diff) < 30.0:
        #         reward += heading_efficiency * 3.0
        #     elif abs(heading_diff) > 90.0:
        #         reward -= 10.0
        
        # # Dynamic collision avoidance rewards
        # is_nmac, nmac_list = self.agent.sensor.get_nmac(self.agent)
        # if is_nmac:
        #     for nmac_uav in nmac_list:
        #         distance = self.agent.current_position.distance(nmac_uav.current_position)
        #         nmac_factor = 1.0 - (distance / self.agent.nmac_radius)
        #         reward -= nmac_factor * 30.0
        
        # # Speed management reward
        # target_speed = self.agent.dynamics.max_speed
        # if current_distance < 1000:
        #     # Reduce target speed when approaching goal
        #     target_speed = max(5.0, self.agent.dynamics.max_speed * (current_distance / 1000))
        # speed_efficiency = 1.0 - abs(current_speed - target_speed) / self.agent.dynamics.max_speed
        # reward += speed_efficiency * 2.0
        
        # # Terminal rewards
        # if current_distance < self.agent.mission_complete_distance:
        #     reward += 1000.0
        
        # return float(reward)
        pass

    def _get_obs(self):

        #FIX: 
        # Depending on the type of constructor used,
        # this method will need to return the correct format of obs data
        """Returns observation of the agent in a specific format (sequential or graph)"""
        if self.obs_space_str == "LSTM-A2C":
            # Get Auto-UAV observation data
            raw_obs = self.agent.get_obs()

            with open("output.txt", "w") as file:
                print(f"Raw observation format: {type(raw_obs)}", file=file)
                print(f"Raw observation content: {raw_obs}", file=file)

            # Transform data for sequential observation format
            transformed_data = transform_sensor_data(
                raw_obs,
                self.max_number_other_agents_observed,
                self.obs_space_str,
                self.sorting_criteria,
            )
            
            return transformed_data
            
        elif self.obs_space_str == "GNN-A2C":
            # Get Auto-UAV observation data
            raw_obs = self.agent.get_obs()
            
            # Transform data for graph observation format
            transformed_data = transform_sensor_data(
                raw_obs, 
                self.max_number_other_agents_observed, 
                self.obs_space_str
            )
            
            return transformed_data
        elif self.obs_space_str == 'UAM_UAV':
            #FIX: test this - chech if this method works and produces correct response
        #   (own_dict, (other_agents, restricted_areas))
            raw_obs =                                       self.agent.get_obs()

            # Transform data for UAM observation format
            transformed_data = transform_sensor_data(
                raw_obs,
                self.max_number_other_agents_observed,
                'UAM_UAV'
            )
            return transformed_data

        else:
            raise RuntimeError(
                "_get_obs \n incorrect self.obs_space_str, check __init__ and self.obs_space_str"
            )

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        # Reset internal state
        #! why seed here (1 of 2)
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            random.seed(seed)
            np.random.seed(seed)
            
        # Reset logger and rendering for new episode
        self.logger.reset()
        self.renderer.reset()

        # Reset tracking for learning agent logging
        self.previous_obs = None
        self.previous_action = None

        # Reset environment state
        self.current_time_step = 0

        #! why seed here (2 of 2)
        random.seed(self._seed)
        np.random.seed(self._seed)
        
        # Create the airspace with restricted areas
        self.airspace = Airspace(self.number_of_vertiport, self.location_name, airspace_tag_list=self.airspace_tag_list)
        
        # Create space, sensors, controllers, and dynamics
        self.atc = ATC(airspace=self.airspace, seed=self._seed)
        
        
        self.map_sensor = MapSensor(airspace=self.airspace, atc=self.atc)
        self.static_controller = StaticController(0, 0)
        self.non_coop_smooth_controller = NonCoopControllerSmooth(10, 2)
        self.non_coop_controller = NonCoopController(10, 1)
        self.pm_dynamics = PointMassDynamics()
        self.agent_pm_dynamics = PointMassDynamics(is_learning=True)

        # Create vertiports
        num_vertiports = min(self.max_vertiports, self.number_of_vertiport)  # Use a reasonable number
        self.airspace.create_n_random_vertiports(num_vertiports)

        # Create UAVs
        num_uavs = min(self.max_uavs, self.number_of_uav)  # Use a reasonable number
        for _ in range(num_uavs):
            self.atc.create_uav(
                UAV_v2,
                controller=self.non_coop_smooth_controller,
                dynamics=self.pm_dynamics,
                sensor=self.map_sensor,
                radius=17,  # Match UAM_UAV parameters
                nmac_radius=150,
                detection_radius=550,
            )

        # Assign start and end points for non-learning UAVs
        for uav in self.atc.get_uav_list():
            start_vertiport = random.choice(self.airspace.get_vertiport_list())
            end_vertiport = random.choice(self.airspace.get_vertiport_list())
            self.atc.assign_vertiport_uav(uav, start_vertiport, end_vertiport)

        #FIX: I think agent should also be developed like UAV -
        #FIX: maybe this is why I previously thought about having two lists
        #FIX: one for UAV and one for Auto UAV 
        # Create learning agent
        self.agent = Auto_UAV_v2(
            dynamics=self.agent_pm_dynamics,
            sensor=self.map_sensor,
            radius=17,
            nmac_radius=150,
            detection_radius=550,
        )

        # Set learning agent in space and assign start/end points
        #FIX: there are methods but, I need to think how to best use them or come up with new ones that feel natural
        self.atc._set_uav(self.agent)
        start_vertiport = random.choice(self.airspace.get_vertiport_list())
        end_vertiport = random.choice(self.airspace.get_vertiport_list())
        self.atc.assign_vertiport_uav(self.agent, start_vertiport, end_vertiport)
        
        # Initialize tracking for reward function
        self.previous_distance = self.agent.current_position.distance(self.agent.end)

        # Get initial observation and info
        obs = self._get_obs()
        info = {}

        # Print debug info for vertiport assignments
        print("--- Vertiport Assignments ---")
        for uav in self.atc.get_uav_list():
            print(f'UAV {uav.id} - Start: {uav.start} end: {uav.end}')
        print(f'Agent - Start: {self.agent.start} end: {self.agent.end}')
        print("---------------------------")

        return obs, info
    
    def render(self):
        """Render the current state of the environment."""
        return self.renderer.render()
    
    def create_animation(self, env_time_step):
        """Create an animation of the environment."""
        return self.renderer.create_animation(env_time_step)
    
    def save_animation(self, animation_obj, file_name):
        """Save animation to file."""
        return self.renderer.save_animation(animation_obj, file_name)

    def close(self):
        """Close the environment and clean up resources."""
        # Close renderer
        self.renderer.close()

        # Close the logger
        self.logger.close()