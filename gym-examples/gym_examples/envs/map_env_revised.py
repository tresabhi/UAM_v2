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
from map_space import MapSpace
from airspace import Airspace
from utils_data_transform import transform_sensor_data

from shapely import Point
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict
from gymnasium import spaces
# from UAV_logger import NonLearningLogger
from map_logger import MapLogger

class MapEnv(gym.Env):
    """
    Urban Air Mobility environment for flying multiple UAVs with an autonomous agent
    in a real-world map with restricted airspace.
    """

    # Maximum number of other agents that can be observed by the learning agent
    max_number_other_agents_observed = 7

   

   

    def __init__(
        self,
        location_name="Austin, Texas, USA",
        airspace_tag_list=[("amenity", "hospital"), ("aeroway", "aerodrome")],
        max_uavs=8,
        max_vertiports=12,
        max_episode_steps=1000,
        max_number_other_agents_observed=7,
        sleep_time=0.005,
        seed=70,
        obs_space_constructor = None,
        obs_space_str=None,
        sorting_criteria=None,
        render_mode=None,
    ):  
        super().__init__()
        
        # Environment configuration
        self.location_name = location_name
        self.airspace_tag_list = airspace_tag_list
        self.max_uavs = max_uavs
        self.max_vertiports = max_vertiports
        self.max_number_other_agents_observed = max_number_other_agents_observed
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

        # Animation data structure
        self.df = pd.DataFrame({
            "current_time_step": [],
            "uav_id": [],
            "uav": [],
            "current_position": [],
            "current_heading": [],
            "final_heading": [],
        })

        # Check observation space configuration
        if obs_space_str == "seq" and sorting_criteria is None:
            raise RuntimeError(
                "env.__init__ needs both obs_space_str and sorting_criteria for sequential observation space"
            )

        # Set observation and action spaces
        #FIX: assign the obs_space_constructor here 
        if self.obs_space_str == "seq":
            self.observation_space = MapEnv.obs_space_seq
        elif self.obs_space_str == "graph":
            self.observation_space = MapEnv.obs_space_graph
        else:
            raise RuntimeError("Choose correct format of obs space: 'seq' or 'graph'")

        self.action_space = spaces.Box(
            low=np.array([-1, -1]),  # [acceleration, heading_change]
            high=np.array([1, 1]),
            shape=(2,),
            dtype=np.float64,
        )

        # Initialize rendering variables
        self.fig = None
        self.ax = None

    def step(self, action):
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
        
        # Save animation data for all UAVs in the environment
        self.add_data(self.agent)
        
        # Check collision with static objects (restricted airspace)
        static_collision, static_collision_info = self.agent.sensor.get_static_collision(self.agent)
        if static_collision:
            print("--- STATIC COLLISION ---")
            print(f"Collision with restricted airspace: {static_collision_info}")
            self.logger.record_collision([self.agent.id], 'static')

            # Clean up trajectory data for collided agent
            if hasattr(self, 'trajectory_by_id') and self.agent.id in self.trajectory_by_id:
                del self.trajectory_by_id[self.agent.id]
            
            return self._get_obs(), -100, False, True, {"collision": True, "type": "static"}
            
        # Check NMAC and collision for learning agent with other UAVs
        is_nmac, nmac_list = self.agent.sensor.get_nmac(self.agent)
        if is_nmac:
            print("--- NMAC ---")
            print(f"NMAC detected: {is_nmac}, and NMAC with {nmac_list}\n")
        
        is_collision, collision_uav_ids = self.agent.sensor.get_collision(self.agent)
        if is_collision:
            print("--- COLLISION ---")
            print(f"Collision detected: {is_collision}, and collision with {collision_uav_ids}\n")
            self.logger.record_collision(collision_uav_ids, 'dynamic')

            # Clean up trajectory data for collided agents
            if hasattr(self, 'trajectory_by_id'):
                for uav_id in collision_uav_ids:
                    if uav_id in self.trajectory_by_id:
                        del self.trajectory_by_id[uav_id]

            self.space.remove_uavs_by_id(collision_uav_ids)
            if len(self.space.uav_list) == 0:
                print("No more UAVs in space")
                return self._get_obs(), -100, False, True, {"collision": True, "type": "dynamic"}
        
        # Now update all non-learning UAVs
        for uav in self.space.get_uav_list():
            if not isinstance(uav, Auto_UAV_v2):  # Only process non-learning UAVs
                # Save animation data
                self.add_data(uav)
                
                # Get non-learning UAV's observations and check for collisions
                observation = uav.get_obs()

                # Check static collisions for non-learning UAVs
                static_collision, static_info = uav.sensor.get_static_collision(uav)
                if static_collision:
                    print(f"--- UAV {uav.id} STATIC COLLISION ---")
                    self.logger.record_collision([uav.id], 'static')

                    # Clean up trajectory for this UAV
                    if hasattr(self, 'trajectory_by_id') and uav.id in self.trajectory_by_id:
                        del self.trajectory_by_id[uav.id]

                    self.space.remove_uavs_by_id([uav.id])
                    continue
                
                # Check NMAC and dynamic collisions
                is_nmac, nmac_list = uav.sensor.get_nmac(uav)
                if is_nmac:
                    print(f"--- UAV {uav.id} NMAC ---")
                
                is_collision, collision_uav_ids = uav.sensor.get_collision(uav)
                if is_collision:
                    print(f"--- UAV {uav.id} COLLISION ---")
                    self.logger.record_collision(collision_uav_ids, 'dynamic')

                    # Clean up trajectory for collided UAVs
                    if hasattr(self, 'trajectory_by_id'):
                        for coll_id in collision_uav_ids:
                            if coll_id in self.trajectory_by_id:
                                del self.trajectory_by_id[coll_id]

                    self.space.remove_uavs_by_id(collision_uav_ids)
                    continue
                
                # Update non-learning UAV's state
                uav_mission_complete_status = uav.get_mission_status()
                uav.set_mission_complete_status(uav_mission_complete_status)

                # If UAV completed its mission, log it and clean up its trajectory
                if uav_mission_complete_status:
                    self.logger.mark_agent_complete(uav.id)
                    if hasattr(self, 'trajectory_by_id') and uav.id in self.trajectory_by_id:
                        del self.trajectory_by_id[uav.id]
                    self.space.remove_uavs_by_id([uav.id])
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
        static_detection, static_info = self.agent.sensor.get_static_detection(self.agent)
        if static_detection:
            info['static_detection'] = True
            info['distance_to_restricted'] = static_info.get('distance', float('inf'))
        
        # Increment time step
        self.current_time_step += 1
        
        return obs, reward, terminated, truncated, info

    def _get_reward(self):
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
        """Returns observation of the agent in a specific format (sequential or graph)"""
        if self.obs_space_str == "seq":
            # Get Auto-UAV observation data
            raw_obs = self.agent.get_obs()
            
            # Add static object detection data
            static_detection, static_info = self.agent.sensor.get_static_detection(self.agent)
            if static_detection:
                raw_obs[0]['static_collision_detected'] = 1
                raw_obs[0]['distance_to_restricted'] = static_info.get('distance', float('inf'))
            else:
                raw_obs[0]['static_collision_detected'] = 0
                raw_obs[0]['distance_to_restricted'] = float('inf')
            
            # Transform data for sequential observation format
            transformed_data = transform_sensor_data(
                raw_obs,
                self.max_number_other_agents_observed,
                "seq",
                self.sorting_criteria,
            )
            
            return transformed_data
            
        elif self.obs_space_str == "graph":
            # Get Auto-UAV observation data
            raw_obs = self.agent.get_obs()
            
            # Add static object detection data
            static_detection, static_info = self.agent.sensor.get_static_detection(self.agent)
            if static_detection:
                raw_obs[0]['static_collision_detected'] = 1
                raw_obs[0]['distance_to_restricted'] = static_info.get('distance', float('inf'))
            else:
                raw_obs[0]['static_collision_detected'] = 0
                raw_obs[0]['distance_to_restricted'] = float('inf')
            
            # Transform data for graph observation format
            transformed_data = transform_sensor_data(
                raw_obs, 
                self.max_number_other_agents_observed, 
                "graph"
            )
            
            return transformed_data
        else:
            raise RuntimeError(
                "_get_obs \n incorrect self.obs_space_str, check __init__ and self.obs_space_str"
            )

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        # Reset internal state
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            random.seed(seed)
            np.random.seed(seed)
            
        # Reset logger for new episode
        self.logger.reset()
        # Reset tracking for learning agent logging
        self.previous_obs = None
        self.previous_action = None

        # Reset animation information for new episode
        self.current_time_step = 0
        self.df = pd.DataFrame({
            "current_time_step": [],
            "uav_id": [],
            "uav": [],
            "current_position": [],
            "current_heading": [],
            "final_heading": [],
        })
        # Reset render history
        self.render_history = []
        # Reset trajectory tracking
        self.trajectory_by_id = {}
        # Clean up animation data if present
        if hasattr(self, 'animation_data'):
            del self.animation_data
        if hasattr(self, 'animation_time_steps'):
            del self.animation_time_steps
        if hasattr(self, 'animation_trajectories'):
            del self.animation_trajectories

        # Create the airspace with restricted areas
        self.airspace = Airspace(self.location_name, airspace_tag_list=self.airspace_tag_list)
        
        # Create space, sensors, controllers, and dynamics
        self.space = MapSpace(
            max_uavs=self.max_uavs, 
            max_vertiports=self.max_vertiports, 
            seed=self._seed,
            airspace=self.airspace
        )
        
        random.seed(self._seed)
        np.random.seed(self._seed)
        
        self.map_sensor = MapSensor(space=self.space)
        self.static_controller = StaticController(0, 0)
        self.non_coop_smooth_controller = NonCoopControllerSmooth(10, 2)
        self.non_coop_controller = NonCoopController(10, 1)
        self.pm_dynamics = PointMassDynamics()
        self.agent_pm_dynamics = PointMassDynamics(is_learning=True)

        # Create vertiports
        num_vertiports = min(self.max_vertiports, 8)  # Use a reasonable number
        self.space.create_random_vertiports(num_vertiports)

        # Create UAVs
        num_uavs = min(self.max_uavs, 4)  # Use a reasonable number
        self.space.create_uavs(
            num_uavs,
            UAV_v2,
            has_agent=True,
            controller=self.non_coop_smooth_controller,
            dynamics=self.pm_dynamics,
            sensor=self.map_sensor,
            radius=17,  # Match UAM_UAV parameters
            nmac_radius=150,
            detection_radius=550,
        )

        # Assign start and end points for non-learning UAVs
        self.space.assign_vertiports("random")

        # Create learning agent
        self.agent = Auto_UAV_v2(
            dynamics=self.agent_pm_dynamics,
            sensor=self.map_sensor,
            radius=17,
            nmac_radius=150,
            detection_radius=550,
        )

        # Set learning agent in space and assign start/end points
        self.space.set_uav(self.agent)
        self.space.assign_vertiport_agent(self.agent)
        
        # Initialize tracking for reward function
        self.previous_distance = self.agent.current_position.distance(self.agent.end)

        # Get initial observation and info
        obs = self._get_obs()
        info = {}

        # Print debug info for vertiport assignments
        print("--- Vertiport Assignments ---")
        for uav in self.space.get_uav_list():
            print(f'UAV {uav.id} - Start: {uav.start} end: {uav.end}')
        print(f'Agent - Start: {self.agent.start} end: {self.agent.end}')
        print("---------------------------")

        return obs, info

    def render(self):
        """Render the environment with matplotlib."""
        if self.render_mode is None:
            return
                
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 10))
            plt.ion()  # Interactive mode on
            
        self.ax.clear()
        
        # Draw the airspace and restricted areas
        self.render_static_assets(self.ax)
        
        # Store current positions for trajectory history
        current_positions = []
        current_uav_ids = []
        
        # Draw vertiports with larger markers
        for vertiport in self.space.get_vertiport_list():
            self.ax.plot(vertiport.x, vertiport.y, 'gs', markersize=12)
        
        # Draw learning agent (with dark blue color like in uam_uav.py)
        agent_pos = self.agent.current_position
        current_positions.append((agent_pos.x, agent_pos.y))
        current_uav_ids.append(self.agent.id)
        
        # Agent detection radius
        agent_detection = Circle((agent_pos.x, agent_pos.y),
                            self.agent.detection_radius,
                            fill=False, color='#0278c2', alpha=0.3, linewidth=2)
        self.ax.add_patch(agent_detection)
        
        # Agent NMAC radius
        agent_nmac = Circle((agent_pos.x, agent_pos.y),
                        self.agent.nmac_radius,
                        fill=False, color='#FF7F50', alpha=0.4, linewidth=2)
        self.ax.add_patch(agent_nmac)
        
        # Agent body
        agent_body = Circle((agent_pos.x, agent_pos.y),
                        self.agent.radius,
                        fill=True, color='#0000A0', alpha=0.9)
        self.ax.add_patch(agent_body)
        
        # Agent heading indicator with thicker line
        heading_length = self.agent.radius * 5  # Make longer for visibility
        dx = heading_length * np.cos(self.agent.current_heading)
        dy = heading_length * np.sin(self.agent.current_heading)
        agent_arrow = FancyArrowPatch((agent_pos.x, agent_pos.y),
                                (agent_pos.x + dx, agent_pos.y + dy),
                                color='black',
                                arrowstyle='->',
                                mutation_scale=10,
                                linewidth=2.5)
        self.ax.add_patch(agent_arrow)
        
        # Agent start-end connection with thicker line
        self.ax.plot([self.agent.start.x, self.agent.end.x],
                    [self.agent.start.y, self.agent.end.y],
                    'b--', alpha=0.6, linewidth=2.0)
        
        # Draw non-learning UAVs
        for uav in self.space.get_uav_list():
            if isinstance(uav, Auto_UAV_v2):
                continue
                
            pos = uav.current_position
            current_positions.append((pos.x, pos.y))
            current_uav_ids.append(uav.id)
            
            # UAV detection radius
            detection = Circle((pos.x, pos.y), uav.detection_radius, 
                        fill=False, color='green', alpha=0.3, linewidth=2)
            self.ax.add_patch(detection)
            
            # UAV NMAC radius
            nmac = Circle((pos.x, pos.y), uav.nmac_radius, 
                    fill=False, color='orange', alpha=0.4, linewidth=2)
            self.ax.add_patch(nmac)
            
            # UAV body
            body = Circle((pos.x, pos.y), uav.radius, 
                    fill=True, color='blue', alpha=0.7)
            self.ax.add_patch(body)
            
            # UAV heading indicator with thicker line
            heading_length = uav.radius * 5  # Make longer for visibility
            dx = heading_length * np.cos(uav.current_heading)
            dy = heading_length * np.sin(uav.current_heading)
            arrow = FancyArrowPatch((pos.x, pos.y),
                                (pos.x + dx, pos.y + dy),
                                color='black',
                                arrowstyle='->',
                                mutation_scale=10,
                                linewidth=2.5)
            self.ax.add_patch(arrow)
            
            # UAV start-end connection with thicker line
            self.ax.plot([uav.start.x, uav.end.x],
                        [uav.start.y, uav.end.y],
                        'g--', alpha=0.6, linewidth=2.0)
        
        # Update trajectory history - maintaining correct UAV ID mapping
        # Create a dictionary of ID to position if it doesn't exist
        if not hasattr(self, 'trajectory_by_id'):
            self.trajectory_by_id = {}
            
        # Add current positions to trajectories
        for uav_id, pos in zip(current_uav_ids, current_positions):
            if uav_id not in self.trajectory_by_id:
                self.trajectory_by_id[uav_id] = []
            self.trajectory_by_id[uav_id].append(pos)
        
        # Draw trajectory lines - only for UAVs that still exist
        for uav_id in current_uav_ids:
            if uav_id in self.trajectory_by_id and len(self.trajectory_by_id[uav_id]) > 1:
                xs, ys = zip(*self.trajectory_by_id[uav_id])
                # Use thicker lines for trajectories
                line_color = '#0000A0' if uav_id == self.agent.id else 'blue'
                self.ax.plot(xs, ys, '-', linewidth=2.5, alpha=0.6, color=line_color)
        
        # Calculate proper plot limits to see the whole map
        # First get vertiport boundaries
        vp_x_coords = [v.x for v in self.space.get_vertiport_list()]
        vp_y_coords = [v.y for v in self.space.get_vertiport_list()]
        
        # Then get UAV positions
        uav_x_coords = [pos[0] for pos in current_positions]
        uav_y_coords = [pos[1] for pos in current_positions]
        
        # Combine to get full area
        all_x_coords = vp_x_coords + uav_x_coords
        all_y_coords = vp_y_coords + uav_y_coords
        
        # Add restricted areas dimensions
        for tag_value in self.airspace.location_tags.keys():
            # Get bounds of restricted areas
            restricted_bounds = self.airspace.location_utm[tag_value].bounds
            if len(restricted_bounds) > 0:
                for bound in restricted_bounds.values:
                    if len(bound) >= 4:  # minx, miny, maxx, maxy
                        all_x_coords.extend([bound[0], bound[2]])
                        all_y_coords.extend([bound[1], bound[3]])
        
        # Set limits with margin
        if all_x_coords and all_y_coords:
            x_min, x_max = min(all_x_coords), max(all_x_coords)
            y_min, y_max = min(all_y_coords), max(all_y_coords)
            
            # Add margin to ensure all elements are visible
            margin = max(500, (x_max - x_min) * 0.1)
            self.ax.set_xlim(x_min - margin, x_max + margin)
            self.ax.set_ylim(y_min - margin, y_max + margin)
        
        self.ax.set_title(f'UAM Simulation - {self.location_name} - Step {self.current_time_step}')
        self.ax.set_aspect('equal')
        
        plt.draw()
        plt.pause(self.sleep_time)

    def create_animation(self, env_time_step):
        """Create an animation of the environment."""
        # Ensure data exists
        if len(self.df) == 0:
            print("No animation data available")
            return None
        
        try:
            from matplotlib.animation import FuncAnimation
            
            # Pre-process data for animation to avoid df lookups during animation
            self.animation_data = {}
            self.animation_time_steps = sorted(self.df['current_time_step'].unique())
            
            # Group data by time step
            for time_step in self.animation_time_steps:
                step_data = self.df[self.df['current_time_step'] == time_step]
                self.animation_data[time_step] = []
                
                # Extract and store UAV data for this time step
                for _, row in step_data.iterrows():
                    uav_data = {
                        'id': row['uav_id'],
                        'position': row['current_position'],
                        'heading': row['current_heading'],
                        'final_heading': row['final_heading'],
                        'is_auto': isinstance(row['uav'], Auto_UAV_v2),
                        'uav': row['uav']
                    }
                    self.animation_data[time_step].append(uav_data)
            
            # Create trajectories by UAV ID
            self.animation_trajectories = {}
            for time_step in self.animation_time_steps:
                for uav_data in self.animation_data[time_step]:
                    uav_id = uav_data['id']
                    position = uav_data['position']
                    
                    if uav_id not in self.animation_trajectories:
                        self.animation_trajectories[uav_id] = []
                    
                    # Store time step and position
                    self.animation_trajectories[uav_id].append((time_step, position))
                    
            # Create a new figure specifically for animation
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Function to update animation frame
            def animate(frame_index):
                if frame_index >= len(self.animation_time_steps):
                    return []
                    
                time_step = self.animation_time_steps[frame_index]
                ax.clear()
                
                # Draw static assets
                self.render_static_assets(ax)
                
                # Draw vertiports with larger markers
                for vertiport in self.space.get_vertiport_list():
                    ax.plot(vertiport.x, vertiport.y, 'gs', markersize=12)
                
                # Draw UAVs at this time step
                current_uav_ids = []
                for uav_data in self.animation_data[time_step]:
                    pos = uav_data['position']
                    heading = uav_data['heading']
                    final_heading = uav_data['final_heading']
                    is_auto = uav_data['is_auto']
                    uav_id = uav_data['id']
                    uav_obj = uav_data['uav']
                    
                    current_uav_ids.append(uav_id)
                    
                    # Set colors based on UAV type
                    detection_color = '#0278c2' if is_auto else 'green'
                    nmac_color = '#FF7F50' if is_auto else 'orange'
                    body_color = '#0000A0' if is_auto else 'blue'
                    
                    # Draw detection radius
                    detection_circle = Circle((pos.x, pos.y),
                                        uav_obj.detection_radius,
                                        fill=False, color=detection_color, alpha=0.3, linewidth=2)
                    ax.add_patch(detection_circle)
                    
                    # Draw NMAC radius
                    nmac_circle = Circle((pos.x, pos.y),
                                    uav_obj.nmac_radius,
                                    fill=False, color=nmac_color, alpha=0.4, linewidth=2)
                    ax.add_patch(nmac_circle)
                    
                    # Draw UAV body
                    body_circle = Circle((pos.x, pos.y),
                                    uav_obj.radius,
                                    fill=True, color=body_color, alpha=0.7)
                    ax.add_patch(body_circle)
                    
                    # Draw current heading arrow
                    heading_length = uav_obj.radius * 5  # Make longer for visibility
                    dx = heading_length * np.cos(heading)
                    dy = heading_length * np.sin(heading)
                    arrow = FancyArrowPatch((pos.x, pos.y),
                                    (pos.x + dx, pos.y + dy),
                                    color='black',
                                    arrowstyle='->',
                                    mutation_scale=10,
                                    linewidth=2.5)
                    ax.add_patch(arrow)
                    
                    # Draw final heading (reference direction) with thicker line
                    ref_length = uav_obj.radius * 4
                    dx_ref = ref_length * np.cos(final_heading)
                    dy_ref = ref_length * np.sin(final_heading)
                    ref_color = 'purple' if is_auto else 'red'
                    ref_arrow = FancyArrowPatch((pos.x, pos.y),
                                        (pos.x + dx_ref, pos.y + dy_ref),
                                        color=ref_color,
                                        arrowstyle='->',
                                        mutation_scale=7.5,
                                        linewidth=2,
                                        alpha=0.6)
                    ax.add_patch(ref_arrow)
                    
                    # Draw start-end connection with thicker line
                    if hasattr(uav_obj, 'start') and hasattr(uav_obj, 'end'):
                        line_color = 'blue' if is_auto else 'green'
                        ax.plot([uav_obj.start.x, uav_obj.end.x],
                            [uav_obj.start.y, uav_obj.end.y],
                            '--', color=line_color, alpha=0.6, linewidth=2.0)
                
                # Draw trajectories up to this time step
                # This ensures we don't show trajectories for removed UAVs
                for uav_id in current_uav_ids:
                    if uav_id in self.animation_trajectories:
                        # Get trajectory points up to current time step
                        traj_points = [(t, p) for t, p in self.animation_trajectories[uav_id] if t <= time_step]
                        
                        if len(traj_points) > 1:
                            # Extract positions
                            positions = [p for _, p in traj_points]
                            xs = [p.x for p in positions]
                            ys = [p.y for p in positions]
                            
                            # Determine color based on UAV type
                            is_auto = any(d['is_auto'] for d in self.animation_data[time_step] if d['id'] == uav_id)
                            line_color = '#0000A0' if is_auto else 'blue'
                            
                            # Draw trajectory line with thicker width
                            ax.plot(xs, ys, '-', linewidth=2.5, alpha=0.6, color=line_color)
                
                # Calculate proper plot limits to see the whole map
                vp_x_coords = [v.x for v in self.space.get_vertiport_list()]
                vp_y_coords = [v.y for v in self.space.get_vertiport_list()]
                
                # Add UAV positions
                uav_x_coords = [uav_data['position'].x for uav_data in self.animation_data[time_step]]
                uav_y_coords = [uav_data['position'].y for uav_data in self.animation_data[time_step]]
                
                # Combine for full area
                all_x_coords = vp_x_coords + uav_x_coords
                all_y_coords = vp_y_coords + uav_y_coords
                
                # Add restricted areas dimensions
                for tag_value in self.airspace.location_tags.keys():
                    # Get bounds of restricted areas
                    restricted_bounds = self.airspace.location_utm[tag_value].bounds
                    if len(restricted_bounds) > 0:
                        for bound in restricted_bounds.values:
                            if len(bound) >= 4:  # minx, miny, maxx, maxy
                                all_x_coords.extend([bound[0], bound[2]])
                                all_y_coords.extend([bound[1], bound[3]])
                
                # Set limits with margin
                if all_x_coords and all_y_coords:
                    x_min, x_max = min(all_x_coords), max(all_x_coords)
                    y_min, y_max = min(all_y_coords), max(all_y_coords)
                    
                    # Add margin to ensure all elements are visible
                    margin = max(500, (x_max - x_min) * 0.1)
                    ax.set_xlim(x_min - margin, x_max + margin)
                    ax.set_ylim(y_min - margin, y_max + margin)
                
                ax.set_title(f'UAM Simulation - Step {time_step}')
                ax.set_aspect('equal')
                
                return []
            
            # Create animation object
            frames = min(env_time_step, len(self.animation_time_steps))
            if frames == 0:
                print("No frames to animate")
                return None
                
            ani = FuncAnimation(
                fig, 
                animate, 
                frames=frames,
                interval=200,
                blit=False
            )
            
            return ani
            
        except Exception as e:
            print(f"Error creating animation: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_animation(self, animation_obj, file_name):
        """Save animation to a file with optimized quality and compatibility."""
        if animation_obj is None:
            print("No animation to save")
            return
            
        try:
            # Save as MP4 first with optimized settings
            try:
                print(f"Saving animation to {file_name}.mp4...")
                from matplotlib.animation import FFMpegWriter
                
                # Try to use a writer with optimized settings
                writer = FFMpegWriter(
                    fps=10,  # Higher fps for smoother playback
                    metadata=dict(title='UAM Simulation'),
                    bitrate=5000,  # Higher bitrate for better quality
                    extra_args=['-vcodec', 'mpeg4', 
                            '-pix_fmt', 'yuv420p',
                            '-q:v', '3']  # Quality value - lower is better quality (1-31)
                )
                
                # First attempt with quality settings
                try:
                    animation_obj.save(
                        f"{file_name}.mp4",
                        writer=writer,
                        dpi=200  # Higher DPI for better quality
                    )
                    print("MP4 saved successfully with high quality settings!")
                except Exception as e:
                    print(f"High quality MP4 save failed: {e}")
                    
                    # Fallback to simpler settings
                    try:
                        print("Trying with simpler MP4 settings...")
                        animation_obj.save(
                            f"{file_name}.mp4",
                            writer='ffmpeg',
                            fps=8,
                            dpi=150
                        )
                        print("MP4 saved successfully with basic settings!")
                    except Exception as e:
                        print(f"Basic MP4 save failed: {e}")
                        
                        # Try with minimal settings
                        try:
                            print("Trying with minimal MP4 settings...")
                            animation_obj.save(
                                f"{file_name}.mp4",
                                writer='ffmpeg',
                                fps=5,
                                dpi=100
                            )
                            print("MP4 saved successfully with minimal settings!")
                        except Exception as e:
                            print(f"Minimal MP4 save failed: {e}")
            except Exception as mp4_error:
                print(f"MP4 saving failed completely: {mp4_error}")
                
            # Save as GIF (as backup)
            try:
                print(f"Saving animation to {file_name}.gif...")
                from matplotlib.animation import PillowWriter
                
                # Use higher quality settings for GIF
                animation_obj.save(
                    f"{file_name}.gif",
                    writer=PillowWriter(fps=8),
                    dpi=150  # Higher DPI for better quality
                )
                print("GIF saved successfully!")
            except Exception as gif_error:
                print(f"GIF save failed: {gif_error}")
                
                # Try with minimal settings
                try:
                    print("Trying with minimal GIF settings...")
                    animation_obj.save(
                        f"{file_name}.gif",
                        writer=PillowWriter(fps=5),
                        dpi=100
                    )
                    print("GIF saved successfully with minimal settings!")
                except Exception as e:
                    print(f"Minimal GIF save failed: {e}")
                    
        except Exception as e:
            print(f"Error in animation saving: {e}")
            import traceback
            traceback.print_exc()

    def add_data(self, uav):
        """Add UAV data to animation dataframe."""
        self.df = self.df._append(
            {
                "current_time_step": self.current_time_step,
                "uav_id": uav.id,
                "uav": uav,
                "current_position": uav.current_position,
                "current_heading": uav.current_heading,
                "final_heading": math.atan2(uav.end.y - uav.current_position.y, 
                                        uav.end.x - uav.current_position.x),
            },
            ignore_index=True,
        )

    def render_static_assets(self, ax):
        """Render the static assets of the environment (map, restricted areas)."""
        # Draw map boundaries
        self.airspace.location_utm_gdf.plot(ax=ax, color="gray", linewidth=0.6)
        
        # Draw restricted areas
        for tag_value in self.airspace.location_tags.keys():
            # Draw actual restricted areas
            self.airspace.location_utm[tag_value].plot(ax=ax, color="red", alpha=0.7)
            # Draw buffer zones
            self.airspace.location_utm_buffer[tag_value].plot(ax=ax, color="orange", alpha=0.3)
        
        # Draw vertiports
        vertiport_points = [v for v in self.space.get_vertiport_list()]
        if vertiport_points:
            gpd.GeoSeries(vertiport_points).plot(ax=ax, color="black", markersize=10)

    def close(self):
        """Close the environment and clean up resources."""
        # close any ongoing animations 
        plt.close('all')

        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        
        # Clean up render and animation resources
        self.render_history = []
        
        if hasattr(self, 'trajectory_by_id'):
            self.trajectory_by_id = {}
        
        if hasattr(self, 'animation_data'):
            del self.animation_data
        
        if hasattr(self, 'animation_time_steps'):
            del self.animation_time_steps
        
        if hasattr(self, 'animation_trajectories'):
            del self.animation_trajectories
        
        if hasattr(self, 'df'):
            self.df = pd.DataFrame({
                "current_time_step": [],
                "uav_id": [],
                "uav": [],
                "current_position": [],
                "current_heading": [],
                "final_heading": [],
            })

        # Close the logger
        self.logger.close()