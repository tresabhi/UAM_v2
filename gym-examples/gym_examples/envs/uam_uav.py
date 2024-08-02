"""
Urban Air Mobility env
-----------------------

This is a custom gym environment, that has been developed for training and testing single agent.
The single agent is defined using the auto_uav variable in this script.

Description of the environment:
-------------------------------

The objective of the environment is to create an airspace where our autonomous uav will 'fly'.
Autonomous UAV will be provided a start and end point, called vertiports, its objective is to traverse from its start vertiport to its end vertiport.
While traversing, it has to avoid 'restricted airspace' and other UAVs.

We define an airspace using a location name. Then we sample vertiports within from our airspace. 
After, we create UAVs which fly using hardcoded policy. An instance of autonomous uav is deployed. 
"""

"""
Side Note:
----------
    If anyone wishes to improve/change the environment I suggest once the assets have been created/updated test them with main.py.
    There are simulator/simulator_basic.py scripts which define an environment with all necessary assets.
    Any improvement/change should be tested using instance of simultor before implementing in custom gym env.

    When we create the UAM env(subclass of gymEnv) it will build an instance that is similar to simulator.
    The initializer arguments of UAM_Env are similar to the simulator, that is location_name, basic_uav_no, vertiport_no, and Auto_uav(only one for single agent env)
    
    Reason for similarity 
    ---------------------
    The simulator scripts create environments where we test functionality of assets. 
    All assets are pulled in to create an environment similar to this one, only auto_uav is not part of simulator.
    Any form of changes to assets should first be tested by using instance of simulator, then added accordingly to this environment. 
    This process will make debugging easier.     
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from typing import List
import time
from geopandas import GeoSeries
import geopandas as gpd
import gymnasium as gym
from gymnasium import spaces

from airspace import Airspace
from airtrafficcontroller import ATC
from uav_basic import UAVBasic
from autonomous_uav import AutonomousUAV
from vertiport import Vertiport


# TODO #10 - Add function signature and description
class UamUavEnv(gym.Env):
    metadata = {"render_mode": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        location_name: str,
        num_vertiport: int,
        num_basic_uav: int,
        sleep_time: float = 0.005,  # sleep time between render frames
        render_mode: str = None,  #! check where this argument is used
    ) -> None:
        """
        Initalizes the UAMUAV environment

        Args:
            location_name (str): Location of the simulation ie. Austin, Texas, USA
            num_vertiports(int): Number of vertiports to generate
            num_basic_uav (int): Number of UAVs to put in the simulation
            sleep_time (float): Time to sleep in between render frames
            render_mode (str): The render mode of the simulation
        """
        # Environment attributes
        self.current_time_step = 0  #! not being used during step
        self.num_vertiports = num_vertiport
        self.num_basic_uavs = num_basic_uav
        self.sleep_time = sleep_time
        self.airspace = Airspace(location_name)
        self.atc = ATC(airspace=self.airspace)

        # Vertiport initialization
        self.atc.create_n_random_vertiports(num_vertiport)

        # UAV initialization
        self.atc.create_n_basic_uavs(num_basic_uav)

        # Environment data
        vertiports_point_array = [
            vertiport.location for vertiport in self.atc.vertiports_in_airspace
        ]
        self.sim_vertiports_point_array = vertiports_point_array
        self.uav_basic_list: List[UAVBasic] = self.atc.basic_uav_list

        # Auto UAV initialization
        start_vertiport_auto_uav = self.get_start_vertiport_auto_uav()
        end_vertiport_auto_uav = self.get_end_vertiport_auto_uav(
            start_vertiport_auto_uav
        )
        self.auto_uav = AutonomousUAV(start_vertiport_auto_uav, end_vertiport_auto_uav)

        # Environment spaces
        self.observation_space = spaces.Dict(
            {
                # agent ID as integer
                "agent_id": spaces.Box(
                    low=0,
                    high=np.iinfo(np.int64).max,
                    shape=(1,),
                    dtype=np.int64,  #! find if it is possible to create ids that take less space
                ),
                # agent speed
                "agent_speed": spaces.Box(  #!need to rename velocity -> speed
                    low=-self.auto_uav.max_speed,  # agent's speed #! need to check why this is negative
                    high=self.auto_uav.max_speed,
                    shape=(1,),
                    dtype=np.float64,
                ),
                # agent deviation
                "agent_deviation": spaces.Box(
                    low=-360,
                    high=360,
                    shape=(1,),
                    dtype=np.float64,  # agent's heading deviation #!should this be -180 to 180, if yes then this needs to be corrected to -180 to 180
                ),
                # intruder detection
                "intruder_detected": spaces.Discrete(
                    2  # 0 for no intruder, 1 for intruder detected
                ),
                # intruder id
                "intruder_id": spaces.Box(
                    low=0,
                    high=np.iinfo(np.int64).max,
                    shape=(1,),
                    dtype=np.int64,  #! find if it is possible to create ids that take less space
                ),
                # distance to intruder
                "distance_to_intruder": spaces.Box(
                    low=0,
                    high=self.auto_uav.detection_radius,
                    shape=(1,),
                    dtype=np.float64,
                ),
                # Relative heading of intruder #!should this be corrected to -180 to 180,
                "relative_heading_intruder": spaces.Box(
                    low=-360, high=360, shape=(1,), dtype=np.float64
                ),
                "intruder_current_heading": spaces.Box(
                    low=-180, high=180, shape=(1,), dtype=np.float64
                ),  # Intruder's heading
            }
        )

        # Normalized action space
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float64,  #! should action choosen form action space be converted back when applied in step()
        )

    def get_vertiport_from_atc(self) -> None:
        """This is a convinience method, for reset()"""

        vertiports_point_array = [
            vertiport.location for vertiport in self.atc.vertiports_in_airspace
        ]
        self.sim_vertiports_point_array = vertiports_point_array

    def get_uav_list_from_atc(self) -> None:
        """This is a convinience method, for reset()"""

        self.uav_basic_list = self.atc.basic_uav_list

    def reset(self, seed: int = None, options: dict = None) -> tuple[dict, dict]:
        """
        resets the environment

        Args:
            seed (int): A number coorelated to the sequnce of randomy generated numbers. (Not in use)
            options (dict): (Not in use)

        Returns:
            observations(dict): Stores all of the agents
            infos(dict): Stores all of the agents
        """

        # TODO #6 - When environment is reset, it should use a seed to reset from. Currently, reset does not use a seed
        super().reset(seed=seed)
        self.np_random, seed = gym.utils.seeding.np_random(seed)

        self.current_time_step = 0
        self.atc.basic_uav_list = []
        self.atc.vertiports_in_airspace = []
        self.uav_basic_list = []
        self.sim_vertiports_point_array = []

        self.atc.create_n_random_vertiports(
            self.num_vertiports
        )  # TODO #7 - all these six methods below needs an argument seed
        self.atc.create_n_basic_uavs(self.num_basic_uavs)
        self.get_vertiport_from_atc()
        self.get_uav_list_from_atc()

        # reset procedure for auto_uav
        self.auto_uav = None

        start_vertiport_auto_uav = (
            self.get_start_vertiport_auto_uav()  # go through all the vertiports
        )

        end_vertiport_auto_uav = self.get_end_vertiport_auto_uav(
            start_vertiport_auto_uav
        )

        self.auto_uav = AutonomousUAV(start_vertiport_auto_uav, end_vertiport_auto_uav)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def get_agent_speed(
        self,
    ) -> np.ndarray:
        return np.array([self.auto_uav.current_speed])

    def get_agent_deviation(
        self,
    ) -> np.ndarray:
        #! should this be converted between -180 to 180
        return np.array(
            [
                self.auto_uav.current_heading_deg
                - self.auto_uav.current_ref_final_heading_deg
            ]
        )

    def _get_obs(self) -> dict:
        agent_id = np.array([self.auto_uav.id])
        agent_speed = self.get_agent_speed()
        agent_deviation = self.get_agent_deviation()
        intruder_info = self.auto_uav.get_state_dynamic_obj(
            self.uav_basic_list, "nmac"
        )  # self.nmac_with_dynamic_obj()
        """
        Gets the observation space of the agent

        Args:
            agent_id (str): The ID of the target UAV

        Returns:
            obs (dict): The observation space of the agent
                agent_id
                agent_speed
                agent_deviation
                intruder_detected
                intruder_id
                distance_to_intruder
                relative_heading_intruder
                intruder_heading
        """

        # TODO #9 - create simple logging to check all observations are in correct format and range- use print()
        if intruder_info:
            intruder_detected = 1
            intruder_id = np.array([intruder_info["intruder_id"]])
            intruder_pos = intruder_info["intruder_pos"]
            intruder_heading = np.array([intruder_info["intruder_current_heading"]])
            distance_to_intruder = np.array(
                [self.auto_uav.current_position.distance(intruder_pos)]
            )
            relative_heading_intruder = np.array(
                [self.auto_uav.current_heading_deg - float(intruder_heading)]
            )
        else:
            intruder_detected = 0
            intruder_id = np.array([0])
            distance_to_intruder = np.array([0])
            relative_heading_intruder = np.array([0])
            intruder_heading = np.array([0])

        #!restricted airspace
        #!implementatation will require updating observation space in __init__
        restricted_airspace, _ = self.auto_uav.get_state_static_obj(
            self.airspace.location_utm_hospital.geometry,
            "detection",  # the collision string represents that we are using detection as indicator
        )
        # what is the output of this method
        # what if there are more than one restricted airspace near auto_uav
        # how do I hypothesize the auto_uav will resolve multiple airspace

        #! restricted airspace detected - should this be detection/nmac
        # if the detection area of uav intersects with a building's detection area,
        # we should do the following -
        # 1) if intersection is detected for 'detection' argument ->
        #                   there should be a small penalty
        #                   based on distance between the uav_footprint and the actual building
        # 2) if there is no building detected the penalty should be zero
        # 3) just like intruder_detected in obs
        #    there will be restricted_airspace detected in obs
        # 4) restricted_airspace will have 0 for not detected 1 for detected,
        #    and if detected - distance will be added to the obs, if not detected distance is leave at 0,
        #    this will be handled by the reward function collect obs and based on restricted_airspace (y/n)
        #    penatly is something or 0.
        if restricted_airspace:
            # what information am i interested in collecting
            distance_to_static_obj_polygon = None
        else:
            distance_to_static_obj_polygon = 0

        observation = {
            "agent_id": agent_id,
            "agent_speed": agent_speed,
            "agent_deviation": agent_deviation,
            "intruder_detected": intruder_detected,
            "intruder_id": intruder_id,
            "distance_to_intruder": distance_to_intruder,
            "relative_heading_intruder": relative_heading_intruder,
            "intruder_current_heading": intruder_heading,
        }

        return observation

    def _get_info(
        self,
    ) -> dict:
        """
        Gets the distance between the uav and target vertiport

        Args:
            agent_id (str): The ID of the target UAV

        Returns:
            (dict): distance to target vertiport
        """

        return {
            "distance_to_end_vertiport": self.auto_uav.current_position.distance(
                self.auto_uav.end_point
            )
        }

    #!rename method for clarity -> this method is for uav_basic in environment
    def set_uav_basic_intruder_list(self) -> list[UAVBasic]:
        """Each UAV needs access to UAV list in the environment
        to perform dynamic detection and collision operation, this method
        assigns the uav_list to all uavs"""

        for uav in self.uav_basic_list:
            uav.get_intruder_uav_list(self.uav_basic_list)

    #!rename method for clarity -> this method is for uav_basic in environment
    def set_uav_basic_building_gdf(self) -> None:
        """Each UAV needs to have information about restriced airspace,
        to perform static detection and collision operation. This setter method
          assigns environment information to all uavs"""

        for uav in self.uav_basic_list:
            uav.get_airspace_building_list(self.airspace.location_utm_hospital_buffer)

    #! WHAT IS THIS
    # def set_auto_uav_building_gdf(self):
    #     #! might need to set building property for auto_uav
    #     # self.auto_uav.get_airspace_building_list(self.airspace.location_utm_hospital_buffer)
    #     self.auto_uav.get

    def step(self, action: tuple) -> tuple[dict, float, bool, bool, dict]:
        """
        This method is used to step the environment, it will step the environment by one timestep.

        The action argument - will be passed to auto_uav's step method

        basic UAVs will step without action. so I will need to modify basic uav_basic in such a way that they will step without action.
        This tells me that basic uav_basic will need to have collision avoidance built into the uav_basic module, such that they can step without action.

        Args:
            action (tuple): the action of the auto_uav

        Return:
            observations (dict): Feeds the state space of the agent
            rewards (int): The reward earned durring the step
            terminations (bool): Checks if the agent has reached its destination
            truncations (bool): Checks for collisions with static and dynamic objects
            infos (dict): distance of agent from target vertiport
        """
        # decomposing action tuple
        # TODO #11 - should acceleration and heading_correction be transformed from normalized value to absolute value
        acceleration = action[0]
        heading_correction = action[1]

        self.set_uav_basic_intruder_list()
        self.set_uav_basic_building_gdf()

        # for uav_basic in uav_basic_list step all uav_basic
        for uav_basic in self.uav_basic_list:
            self.atc.has_left_start_vertiport(uav_basic)
            self.atc.has_reached_end_vertiport(uav_basic)
            uav_basic.step()

        # Auto_uav step
        self.auto_uav.step(acceleration, heading_correction)

        #! WE DO NOT NEED TO PERFORM HAS_LEFT_START_VERTIPORT and HAS_REACHED_END_VERTIPORT
        #! because once the auto uav reaches its end_vertiport the training stops and we reset the environment
        # TODO #12 - check if environment should be reset after completing only one journey
        obs = self._get_obs()
        reward = self.get_reward(obs)
        info = self._get_info()

        # TODO - develop methods for termination and truncation
        # Logic for termination and truncation
        auto_uav_current_position = self.auto_uav.current_position
        auto_uav_end_position = self.auto_uav.end_point

        distance_to_end_point = auto_uav_current_position.distance(
            auto_uav_end_position
        )

        """if distance to endpoint is less than landing proximity of auto_uav
                we have reached our end vertiport, thus we can terminate our episode"""
        if distance_to_end_point < self.auto_uav.landing_proximity:
            reached_end_vertiport = True
        else:
            reached_end_vertiport = False

        if reached_end_vertiport:
            terminated = True
        else:
            terminated = False

        # check collision with static object
        collision_static_obj, _ = self.auto_uav.get_state_static_obj(
            self.airspace.location_utm_hospital.geometry,
            "collision",  # self.collision_with_static_obj()
        )

        # check collision with dynamic object
        collision_dynamic_obj = self.auto_uav.get_collision(
            self.uav_basic_list
        )  # self.collision_with_dynamic_obj()

        collision_detected = collision_static_obj or collision_dynamic_obj

        if collision_detected:
            truncated = True
        else:
            truncated = False

        self.current_time_step += 1

        return obs, reward, terminated, truncated, info

    def get_reward(self, obs: dict) -> float:
        """
        Returns the reward the agent earns at each step

        Args:
            obs (dict): The observation space of the agent
                agent_id
                agent_speed
                agent_deviation
                intruder_detected
                intruder_id
                distance_to_intruder
                relative_heading_intruder
                intruder_heading

        Returns:
            reward_sum (float): Reward earned by the agent in that time step
        """

        punishment_existing = -0.1
        if obs["intruder_detected"] == 0:
            punishment_closeness: float = 0.0
        else:
            normed_nmac_distance = (
                self.auto_uav.nmac_radius / self.auto_uav.detection_radius
            )  # what is this and why do i need it
            punishment_closeness = -np.exp(
                (normed_nmac_distance - obs["distance_to_intruder"]) * 10
            )

        reward_to_destination = float(obs["agent_speed"]) * float(
            np.cos(obs["agent_deviation"])
        )

        punishment_deviation = float(-2 * (obs["agent_deviation"] / np.pi) ** 2)

        reward_sum = (
            punishment_existing
            + punishment_closeness
            + punishment_deviation
            + reward_to_destination
        )

        reward_sum *= float(self.current_time_step)

        return float(reward_sum)

    def render_init(
        self,
    ) -> tuple[Figure, Axes]:
        """
        Initalizes the rendering

        Returns:
            fig(plt.Figure): The outside of the graph that is rendered
            ax(plt.Axes): The backdrop of the graph
        """
        fig, ax = plt.subplots()
        return fig, ax

    def render_static_assets(
        self, ax: Axes
    ) -> None:  #! spelling error - fix everywhere this is used
        """
        Renders static assets onto the graph

        Args:
            ax(plt.Axes): The backdrop of the graph
        """
        self.airspace.location_utm_gdf.plot(ax=ax, color="gray", linewidth=0.6)
        self.airspace.location_utm_hospital_buffer.plot(ax=ax, color="red", alpha=0.3)
        self.airspace.location_utm_hospital.plot(ax=ax, color="black")
        # adding vertiports to static plot
        gpd.GeoSeries(self.sim_vertiports_point_array).plot(ax=ax, color="black")

    def render(self, fig: Figure, ax: Axes) -> None:
        """
        Renders everything in the graph

        Args:
            fig(plt.Figure): The outside of the graph that is rendered
            ax(plt.Axes): The backdrop of the graph
        """
        plt.cla()
        self.render_static_assets(ax)

        # uav_basic PLOT LOGIC
        for uav_obj in self.uav_basic_list:
            uav_footprint_poly = uav_obj.uav_polygon_plot(uav_obj.uav_footprint)
            uav_footprint_poly.plot(ax=ax, color=uav_obj.uav_footprint_color, alpha=0.3)

            uav_nmac_poly = uav_obj.uav_polygon_plot(uav_obj.nmac_radius)
            uav_nmac_poly.plot(ax=ax, color=uav_obj.uav_nmac_radius_color, alpha=0.3)

            uav_detection_poly = uav_obj.uav_polygon_plot(uav_obj.detection_radius)
            uav_detection_poly.plot(
                ax=ax, color=uav_obj.uav_detection_radius_color, alpha=0.3
            )
            x_current, y_current, dx_current, dy_current = (
                uav_obj.get_uav_current_heading_arrow()
            )
            ax.arrow(x_current, y_current, dx_current, dy_current, alpha=1)
            x_final, y_final, dx_final, dy_final = uav_obj.get_uav_final_heading_arrow()
            ax.arrow(x_final, y_final, dx_final, dy_final, alpha=0.8)

        auto_uav_footprint_poly = self.auto_uav.uav_polygon_plot(
            self.auto_uav.collision_radius
        )
        auto_uav_footprint_poly.plot(
            ax=ax, color=self.auto_uav.uav_footprint_color, alpha=0.3
        )

        auto_uav_nmac_poly = self.auto_uav.uav_polygon_plot(self.auto_uav.nmac_radius)
        auto_uav_nmac_poly.plot(
            ax=ax, color=self.auto_uav.uav_nmac_radius_color, alpha=0.3
        )

        auto_uav_detection_poly = self.auto_uav.uav_polygon_plot(
            self.auto_uav.detection_radius
        )
        auto_uav_detection_poly.plot(
            ax=ax, color=self.auto_uav.uav_detection_radius_color, alpha=0.3
        )
        auto_x_current, auto_y_current, auto_dx_current, auto_dy_current = (
            self.auto_uav.get_uav_current_heading_arrow()
        )
        ax.arrow(
            auto_x_current, auto_y_current, auto_dx_current, auto_dy_current, alpha=1
        )
        auto_x_final, auto_y_final, auto_dx_final, auto_dy_final = (
            self.auto_uav.get_uav_final_heading_arrow()
        )
        ax.arrow(auto_x_final, auto_y_final, auto_dx_final, auto_dy_final, alpha=0.8)

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(self.sleep_time)

    def get_start_vertiport_auto_uav(
        self,
    ) -> Vertiport:
        """
        Gets the vertiport for the UAV to start at

        Returns:
            start_vertiport_auto_uav (Vertiport): the starting vertiport for the UAV
        """
        for vertiport in self.atc.vertiports_in_airspace:
            if len(vertiport.uav_list) == 0:
                start_vertiport_auto_uav = vertiport
        return start_vertiport_auto_uav

    def get_end_vertiport_auto_uav(self, start_vertiport: Vertiport) -> Vertiport:
        """
        Gets the vertiport for the UAV to target

        Returns:
            scome_vertiport (Vertiport): the target vertiport for the UAV
        """
        some_vertiport = self.atc.provide_vertiport()
        while some_vertiport.location == start_vertiport.location:
            some_vertiport = self.atc.provide_vertiport()
        return some_vertiport

    #! there are UAV methods that accomplish this task - remove this method and use UAV native methods
    def collision_with_static_obj(
        self,
    ) -> bool:
        """
        Checks for a collision between the UAV and static objects (hospitals)

        Returns:
            colsion_with_static_obj (bool): Returns true if there has been a collision with a static object
        """
        collision_with_static_obj, _ = self.auto_uav.get_state_static_obj(
            self.airspace.location_utm_hospital.geometry, "collision"
        )
        return collision_with_static_obj

    #! there are UAV methods that accomplish this task - remove this method and use UAV native methods
    def collision_with_dynamic_obj(
        self,
    ) -> bool:
        collision = self.auto_uav.get_collision(self.uav_basic_list)
        return collision

    #! there are UAV methods that accomplish this task - remove this method and use UAV native methods
    def nmac_with_dynamic_obj(
        self,
    ) -> dict:
        nmac_info_dict = self.auto_uav.get_state_dynamic_obj(
            self.uav_basic_list, "nmac"
        )
        return nmac_info_dict

        # TODO - determine if one run of experiment will end when auto_uav reaches its first destination, or should we define a number of destinations or should it be a number of steps based completion
        #! auto uav_basic will also need these two methods for moving to the next vertiport
        # self.atc.has_left_start_vertiport(uav_basic) -> will need these two depending on how an experiment ends
        # self.atc.has_reached_end_vertiport(uav_basic)

    def _render_frame(
        self,
    ):
        pass

    def close(
        self,
    ):
        pass
