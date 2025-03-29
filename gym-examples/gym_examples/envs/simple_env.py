from uav_v2 import UAV_v2
from auto_uav_v2 import Auto_UAV_v2
from controller_static import StaticController
from controller_non_coop import NonCoopController
from controller_non_coop_smooth import NonCoopControllerSmooth
from controller_non_coop_ORCA import ORCA_controller
from dynamics_orca import ORCA_Dynamics
from dynamics_point_mass import PointMassDynamics
from sensor_universal import UniversalSensor
from space import Space
import math
import numpy as np
from shapely import Point
import shapely
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium import spaces
from utils_data_transform import transform_sensor_data
from UAV_logger import NonLearningLogger

class SimpleEnv(gym.Env):
 
    def __init__(
        self,
        non_learning_uav_count=1,
        obs_space_str=None,
        sorting_criteria=None,
        max_uavs=12,
        max_vertiports=14,
        max_number_other_agents_observed=7,
        seed=42,
    ):  
        '''
        A simple gymnasium env for UAVs in 2D space.
        Includes learning and non-learning agents. 
        Primary goal of this environment is to test collision avoidance models.

        Args: 
            non_learning_uav_count: int - number of non_leanring UAV for this instance of SimpleEnv
            max_uavs: int - total UAVs in the env, 
            max_vertiports: int - vertiports in env, 
            max_number_other_agents_observed: int - for sequence model(LSTM) max number of other_agents in sequence,
            obs_space_str: str - defines the observation space type either sequence(for LSTM) or graph(for GNN variants),
            sorting_criteria: str - used for sorting other_agents' sequence used by model(LSTM)
        Returns: 
            None
        '''
        super().__init__()
        # Define sequential observation space for LSTM
        self._obs_space_seq = spaces.Dict(
            {
                # TWO form of experiment - one with end point, and one without end-point
                "no_other_agents": Box(
                    low=0, high=max_number_other_agents_observed, shape=(1,)
                ),
                #! SUGGESTION: Add end_vertiport co-ord !#
                "dist_goal": Box(low=0, high=250, shape=(1,), dtype=np.float32),
                "heading_ego_frame": Box(low=-180, high=180, shape=(1,), dtype=np.float32),
                "current_speed": Box(low=0, high=25, shape=(1,), dtype=np.float32),
                "radius": Box(low=0, high=20, shape=(1,), dtype=np.float32),  # UAV size
                "other_agents_states": Box(  # p_parall, p_orth, v_parall, v_orth, other_agent_radius, combined_radius, dist_2_other
                    low=np.full(
                        (max_number_other_agents_observed, 7), -np.inf 
                    ),  # Use -inf as the lower bound for unspecified dimensions
                    high=np.full(
                        (max_number_other_agents_observed, 7), np.inf
                    ),  # Use inf as the upper bound for unspecified dimensions
                    shape=(
                        max_number_other_agents_observed,
                        7,
                    ),  # Array size (other_observed_uav, 7)
                    dtype=np.float32,  # Ensure consistent data type
                ),
            }
        )

        # Define graph observsation space for GNN
        self._obs_space_graph = spaces.Dict(
            {
                "num_other_agents": spaces.Box(low=0, high=100, shape=(), dtype=np.int64),
                "agent_dist_to_goal": spaces.Box(
                    low=0, high=np.inf, shape=(), dtype=np.float32
                ),
                "agent_end_point": spaces.Box(
                    low=np.array([-np.inf, -np.inf]),
                    high=np.array([np.inf, np.inf]),
                    shape=(2,),
                    dtype=np.float32,
                ),
                "agent_current_position": spaces.Box(
                    low=np.array([-np.inf, -np.inf]),
                    high=np.array([np.inf, np.inf]),
                    shape=(2,),
                    dtype=np.float32,
                ),
                "graph_feat_matrix": spaces.Box(
                    low=np.full(
                        (max_number_other_agents_observed + 1, 5), -np.inf
                    ),  # +1 for the host UAV
                    high=np.full((max_number_other_agents_observed + 1, 5), np.inf),
                    shape=(max_number_other_agents_observed + 1, 5),
                    dtype=np.float32,
                ),
                "edge_index": spaces.Box(
                    low=0,
                    high=max_number_other_agents_observed,
                    shape=(2, max_number_other_agents_observed),
                    dtype=np.int64,
                ),
                "edge_attr": spaces.Box(
                    low=0,
                    high=np.inf,
                    shape=(
                        max_number_other_agents_observed,
                        1,
                    ),  # Assuming 1 feature per edge (e.g., distance)
                    dtype=np.float32,
                ),
                "mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(max_number_other_agents_observed + 1,),  # +1 for the host UAV
                    dtype=np.float32,
                ),
            }
        )

        # Initialize the non-learning agent logger
        self.non_learning_logger = NonLearningLogger()

        # env obs chech
        if obs_space_str == "seq" and sorting_criteria == None:
            raise RuntimeError(
                "env.__init__ needs both obs_space_str and sorting_criteria for sequential observation space"
            )

        # env needs to initialzed with number of UAVs, vertiports, and some parameters that are needed for space.
        # The parameters will be used by methods from space to create UAVs, vertiports, assign start-end points, etc.
        self.non_learning_uav_count = non_learning_uav_count
        self.max_uavs = max_uavs
        self.max_vertiports = max_vertiports
        self.max_number_other_agents_observed = max_number_other_agents_observed
        self._seed = seed
        self.obs_space_str = obs_space_str
        self.sorting_criteria = sorting_criteria

        # Initialize rendering variables
        self.fig = None
        self.ax = None
        self.render_history = []  # Store positions for trajectory visualization

        if self.obs_space_str == "seq":
            self.observation_space = self._obs_space_seq
        elif self.obs_space_str == "graph":
            self.observation_space = self._obs_space_graph
        else:
            raise RuntimeError(
                "Choose correct format of obs space and provide correct string to init"
            )

        self.action_space = spaces.Box(
            low=np.array([-1, -1]),  # acceleration, heading_change
            high=np.array([1, 1]),
            shape=(2,),
        )

    def step(self, action):
        """
        Execute one time step within the environment.
        
        Args:
            action: Action provided by the learning agent
            
        Returns:
            obs: Observation of the learning agent's state
            reward: Reward signal
            done: Whether the episode has ended
            info: Additional information
        """
        
        #### LEARNING AGENT BLOCK ####
        if __debug__:  # Debug printing for action sampling verification
            # print(f"\nBefore update - Agent position: {self.agent.current_position}")
            # print(f"Action applied: {action}")
            pass
        self.agent.dynamics.update(self.agent, action)
        if __debug__:  # Debug printing for action application verification
            # print(f"After update - Agent position: {self.agent.current_position}")
            pass
        
        # Check NMAC and collision for learning agent
        is_nmac, nmac_list = self.agent.sensor.get_nmac(self.agent)
        if is_nmac:
            print("--- NMAC ---")
            print(f"NMAC detected:{is_nmac}, and NMAC with {nmac_list}\n")
        
        is_collision, collision_uav_ids = self.agent.sensor.get_collision(self.agent)
        if is_collision:
            print("---COLLISION---")
            print(f"Collision detected:{is_collision}, and collision with {collision_uav_ids}\n")
            self.space.remove_uavs_by_id(collision_uav_ids)
            if len(self.space.uav_list) == 0:
                print("NO more uavs in space")
                return self._get_obs(), -100, True, {"collision": True}
        
        #### NON-LEARNING AGENT BLOCK ####
        # Now update all non-learning UAVs
        for uav in self.space.get_uav_list():
            if not isinstance(uav, Auto_UAV_v2):  # Only process non-learning UAVs
                # Get non-learning UAV's observations and check for collisions
                observation = uav.get_obs() # Raw observation data for non-learning controllers

                is_nmac, nmac_list = uav.sensor.get_nmac(uav)
                if is_nmac:
                    print("--- NMAC ---")
                    print(f"NMAC detected:{is_nmac}, and NMAC with {nmac_list}\n")

                is_collision, collision_uav_ids = uav.sensor.get_collision(uav)
                if is_collision:
                    print("---COLLISION---")
                    print(f"Collision detected:{is_collision}, and collision with {collision_uav_ids}\n")
                    # Log final state before collision and end episode
                    self.non_learning_logger.record_collision(collision_uav_ids, 'dynamic')
                    self.space.remove_uavs_by_id(collision_uav_ids)
                    if len(self.space.uav_list) == 0:
                        print("NO more uavs in space")
                        return self._get_obs(), -100, True, {"collision": True}
                
                # Update non-learning UAV's state
                uav_mission_complete_status = uav.get_mission_status()
                uav.set_mission_complete_status(uav_mission_complete_status)
                
                # Get and apply non-learning UAV's action based on its controller
                uav_action = uav.get_action(observation=observation)
                print(f'uav action: {uav_action}')
                uav.dynamics.update(uav, uav_action)

                # Log the state-action pair for this non-learning UAV
                self.non_learning_logger.log_step(uav.id, observation, uav_action)

                # If UAV completed its mission, log it
                if uav_mission_complete_status:
                    self.non_learning_logger.mark_agent_complete(uav.id)
                
                if __debug__:  # Debug printing for state verification
                    # print(f"Non-learning UAV State:\n{uav.get_state()}\n")
                    pass
        
        # Get learning agent's observation and status
        obs = self._get_obs()  # This transforms raw observation into the correct format (seq/graph)
        reward = self._get_reward(action)
        done = self.agent.get_mission_status()  # Get completion status from agent
        info = self._get_info()
        
        return obs, reward, done, info

    def _get_reward(self, action):
        '''
        Reward function for the environment

        Args:
            None #Fix: I think this method should use info from step/reset to determine the reward
        
        Returns:
            None
        '''
        digression = self.agent.dist_to_goal #shapely.distance(self.agent.end, self.agent.current_position) # penalty distance digression
        
        deviation =   self.agent.current_heading - self.agent.get_state()['dest_heading'] # penalty deviation between current_heading, and destination_heading
        
        dest = None       # reward for reaching destination
        
        some_turn_rate = 10
        if action[1] > some_turn_rate:
            turn_rate = None  # penalty for high turn rate 
        else:
            turn_rate = 0
        
        if self.agent.sensor.get_nmac(self.agent)[0]:
            nmac_incidence = True       # penalty for nmac incidence 
            #! how to calculate closeness for multiple other_agents 
            closeness = None  # penalty for closeness to intruder/other_agent
        else: 
            nmac_incidence = False




    def _get_obs(self) -> spaces.Dict:
        """Returns observation of the agent in a specific format"""
        if self.obs_space_str == "seq":
            # get Auto-UAV observation data
            raw_obs = self.agent.get_obs()
            # pass observation to transform_data
            transformed_data = transform_sensor_data(
                raw_obs,
                self.max_number_other_agents_observed,
                "seq",
                self.sorting_criteria,
            )
            # return transformed obs_data
            return transformed_data
        
        elif self.obs_space_str == "graph":
            # get Auto-UAV observation data
            raw_obs = self.agent.get_obs()
            # pass observation to transform_data
            transformed_data = transform_sensor_data(
                raw_obs, 
                self.max_number_other_agents_observed, 
                "graph"
            )
            # return transformed obs_data
            return transformed_data
        
        else:
            raise RuntimeError(
                "_get_obs \n incorrect self.obs_space_str, check __init__ and self.obs_space_str"
            )
    
    
    def _get_info(self):
    #FIX: what should be an ideal return of this method
    # should contain auxilary diagnostic information
    # debugging, logging, analysis
    #  
    # def _get_info(self):
    #     agent_state = self.agent.get_state()
    #     info = {
    #         "agent_position": [agent_state["position"].x, agent_state["position"].y],
    #         "agent_heading": agent_state["heading"],
    #         "agent_speed": agent_state["speed"],
    #         "agent_distance_to_goal": agent_state["distance_to_goal"],
    #         "agent_goal_reached": self.agent.get_mission_status(),
    #         "nmac_occurred": self.agent.sensor.nmac_flag,
    #         "nmac_with_ids": self.agent.sensor.nmac_ids,
    #         "collision_occurred": self.agent.sensor.collision_flag,
    #         "collision_with_ids": self.agent.sensor.collision_ids,
    #         "num_uavs_remaining": len(self.space.get_uav_list()),
    #         "num_vertiports": len(self.space.get_vertiport_list()),
    #         "mission_complete": self.agent.get_mission_status()
    #     }
    #     return info  
        return self.agent.get_state() 

     

    def reset(self):
        # Reset logger for new episode
        self.non_learning_logger.reset()

        # if agent has collision, call reset.
        # if agent reaches goal, call reset.
        # if agent reaches max steps, call reset.
        self.space = Space(
            max_uavs=self.max_uavs, max_vertiports=self.max_vertiports, seed=self._seed
        )
        self.universal_sensor = UniversalSensor(space=self.space)
        self.static_controller = StaticController(0, 0)
        self.non_coop_smooth_controller = NonCoopControllerSmooth(10, 2)
        self.non_coop_controller_orca = ORCA_controller(10,np.pi, 5, 0.1)
        self.non_coop_controller = NonCoopController(10, 1)
        
        self.orca_dynamics = ORCA_Dynamics()
        self.pm_dynamics = PointMassDynamics()
        self.agent_pm_dynamics = PointMassDynamics(is_learning=True)
        self.universal_sensor = UniversalSensor(space=self.space)

        # --- Vertiport construction ---
        self.space.create_circular_pattern_vertiports(8, 300)
        # self.space.create_random_pattern_vertiports(8,300)

        # --- UAV construction ---
        #! create_UAVs method returns 1 less than no. uav 
        self.space.create_uavs(
            self.non_learning_uav_count, #FIX: fix this method so that 
            UAV_v2,
            has_agent=True, #! this attribute reduces the non_leanring agent count by 1 
            controller=self.non_coop_smooth_controller,
            dynamics=self.pm_dynamics,
            sensor=self.universal_sensor,
            radius=5,
            nmac_radius=20,
            detection_radius=50,
        )

        # --- UAV start-end assignment ---
        self.space.assign_vertiports("opposite")

        # --- create Agent (auto UAV) ---
        self.agent = Auto_UAV_v2(
            dynamics=self.agent_pm_dynamics,
            sensor=self.universal_sensor,
            radius=5,
            nmac_radius=20,
            detection_radius=50,
        )

        # ---Agent (Auto UAV) start-end assignment ---
        self.space.set_uav(self.agent)
        self.space.assign_vertiport_agent(self.agent)

        #! _get_obs() is for agent, which will accept obs_str and pass it to all relevant methods of agent that will create its obs.
        obs = self._get_obs()  # self.agent.get_obs()
        info = self._get_info()  # self.agent.get_info()

        # Log initial states of non-learning UAVs
        for uav in self.space.get_uav_list():
            if not isinstance(uav, Auto_UAV_v2):
                observation = uav.get_obs()
                action = uav.get_action(observation=observation)
                self.non_learning_logger.log_step(uav.id, observation, action)

        ##### check UAV and start - end points ######

        for uav in self.space.get_uav_list():
            print(f'Start: {uav.start} end: {uav.end} ')



        ##### check UAV and start - end points ######

        return obs, info

    def render(self):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            plt.ion()
            
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        
        # Draw vertiports
        for vertiport in self.space.get_vertiport_list():
            self.ax.plot(vertiport.x, vertiport.y, 'gs', markersize=10)
        
        # Initialize position history with learning agent first
        current_positions = [(self.agent.current_position.x, self.agent.current_position.y)]
        
        # Draw learning agent first
        agent_pos = self.agent.current_position
        
        # Agent detection and NMAC radius
        agent_detection = Circle((agent_pos.x, agent_pos.y),
                            self.agent.detection_radius,
                            fill=False, color='#4A90E2', alpha=0.3)
        self.ax.add_patch(agent_detection)
        
        agent_nmac = Circle((agent_pos.x, agent_pos.y),
                        self.agent.nmac_radius,
                        fill=False, color='#FF7F50', alpha=0.4)
        self.ax.add_patch(agent_nmac)
        
        # Agent body (dark blue)
        agent_body = Circle((agent_pos.x, agent_pos.y),
                        self.agent.radius,
                        fill=True, color='#0000A0', alpha=0.9)
        self.ax.add_patch(agent_body)
        
        # Agent heading indicator
        heading_length = self.agent.radius * 4
        dx = heading_length * np.cos(self.agent.current_heading)
        dy = heading_length * np.sin(self.agent.current_heading)
        agent_arrow = FancyArrowPatch((agent_pos.x, agent_pos.y),
                                (agent_pos.x + dx, agent_pos.y + dy),
                                color='black',
                                arrowstyle='->',
                                mutation_scale=15)
        self.ax.add_patch(agent_arrow)
        
        # Agent start-end connection
        self.ax.plot([self.agent.start.x, self.agent.end.x],
                    [self.agent.start.y, self.agent.end.y],
                    'b--', alpha=0.3)
        
        # Draw non-learning UAVs
        for uav in self.space.get_uav_list():
            if isinstance(uav, Auto_UAV_v2):
                continue
                
            pos = uav.current_position
            current_positions.append((pos.x, pos.y))
            
            detection = Circle((pos.x, pos.y), uav.detection_radius, 
                        fill=False, color='#4A90E2', alpha=0.3)
            self.ax.add_patch(detection)
            
            nmac = Circle((pos.x, pos.y), uav.nmac_radius, 
                    fill=False, color='#FF7F50', alpha=0.4)
            self.ax.add_patch(nmac)
            
            body = Circle((pos.x, pos.y), uav.radius, 
                    fill=True, color='red', alpha=0.7)
            self.ax.add_patch(body)
            
            heading_length = uav.radius * 4
            dx = heading_length * np.cos(uav.current_heading)
            dy = heading_length * np.sin(uav.current_heading)
            arrow = FancyArrowPatch((pos.x, pos.y),
                                (pos.x + dx, pos.y + dy),
                                color='black',
                                arrowstyle='->',
                                mutation_scale=15)
            self.ax.add_patch(arrow)
            
            self.ax.plot([uav.start.x, uav.end.x],
                        [uav.start.y, uav.end.y],
                        'g--', alpha=0.3)
        
        # Store and draw trajectories with consistent ordering
        self.render_history.append(current_positions)
        if len(self.render_history) > 1:
            for i in range(len(current_positions)):
                trajectory = []
                for step in self.render_history:
                    if i < len(step):  # Only add point if it exists (handles UAV removals)
                        trajectory.append(step[i])
                if trajectory:
                    xs, ys = zip(*trajectory)
                    self.ax.plot(xs, ys, '-o', markersize=2, alpha=0.5)
        
        # Set plot limits and title
        x_coords = [v.x for v in self.space.get_vertiport_list()]
        y_coords = [v.y for v in self.space.get_vertiport_list()]
        margin = 50
        self.ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
        self.ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
        self.ax.set_title('UAM Simulation Environment')
        
        plt.draw()
        plt.pause(0.01)
        
    def close(self):
        """Close the rendering window and logger."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.render_history = []

        # Close the non-learning logger
        self.non_learning_logger.close()

    def seed(self, seed=None):
        pass
    

    def get_obs_shape(self):
        learning_agent_state_shape = 0
        for k in self.observation_space.keys():
            if k != 'other_agents_states' and k != 'mask':
                obs_shape = self.observation_space[k].shape
                learning_agent_state_shape += obs_shape[0]
            else: 
                other_agents_states_shape = self.observation_space[k].shape[1]
        
        shape_dict = dict(learning_agent_state_shape = learning_agent_state_shape, other_agents_states_shape=other_agents_states_shape)
        
        return shape_dict
