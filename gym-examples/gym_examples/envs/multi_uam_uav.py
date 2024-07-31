import functools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from typing import List, Dict
import time
from geopandas import GeoSeries
import geopandas as gpd
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv

from airspace import Airspace
from airtrafficcontroller import ATC
from autonomous_uav import AutonomousUAV
from vertiport import Vertiport


class UamUavEnvPZ(ParallelEnv):
    metadata = {
        "name": "multi_uav_uam_v0",
    }

    def __init__(
        self,
        location_name: str,
        num_vertiports: int,
        num_auto_uav: int,
        sleep_time: float = 0.05,
        render_mode: str = None,
    ) -> None:
        """
        Initalizes multple parallel enviormnets for UamUav

        Args:
            location_name (str): Location of the simulation ie. Austin, Texas, USA
            num_vertiports(int): Number of vertiports to generate
            num_auto_uav (int): Number of UAVs to put in the simulation
            sleep_time (float): Time to sleep in between steps
            render_mode (str): The render mode of the simulation
        """
        # Environment attributes
        self.current_time_step = 0
        self.num_vertiports = num_vertiports
        self.num_auto_uav = num_auto_uav
        self.sleep_time = sleep_time
        self.airspace = Airspace(location_name)
        self.atc = ATC(airspace=self.airspace)

        # AUTO UAV detail
        # TODO #17
        self.auto_uav_max_speed = 43  # self.auto_uavs_list[0].max_speed
        self.auto_uav_detection_radius = 550  # self.auto_uavs_list[0].detection_radius

        # Vertiport Initialization
        self.atc.create_n_random_vertiports(self.num_vertiports)

        # Auto UAV initialization
        self.atc.create_n_auto_uavs(self.num_auto_uav)

        # Environment data
        vertiports_point_array = [
            vertiport.location for vertiport in self.atc.vertiports_in_airspace
        ]
        self.sim_vertiports_point_array = vertiports_point_array

        # make a list of AutoUAV
        self.auto_uavs_list = self.atc.auto_uavs_list

        # make an attribute -> self.auto_uavs_dict = {auto_uav.id:auto_uav for auto_uav in list_AUTO_UAV}
        self.auto_uavs_dict = {
            auto_uav.id: auto_uav for auto_uav in self.auto_uavs_list
        }

        # Petting Zoo API attributes
        self.possible_agents = list(
            self.auto_uavs_dict.keys()
        )  # .keys() because auto_uavs_dict will need to be a dictionary
        self.agents = (
            self.possible_agents
        )  # agents is a list of keys of each auto_uav id
        # self.num_agents = num_auto_uav
        # self.max_num_agents = max_agents

    def has_terminated(self, agent_id: str) -> bool:
        """
        Checks to see if the agent has reached the target vertiport

        Args:
            agent_id (str): The ID of the target UAV

        Returns:
            terminated (bool): True or false. Has the UAV reached the targeted vertiport
        """
        agent = self.auto_uavs_dict[agent_id]
        dist_to_end_point = agent.current_position.distance(agent.end_point)

        if dist_to_end_point < agent.landing_proximity:
            terminated = True
        else:
            terminated = False

        return terminated

    def has_truncated(self, agent_id: str) -> bool:
        """
        Checks for collision between UAV and static or dynamic object

        Args:
            agent_id (str): The ID of the target UAV

        Returns:
            truncated (bool): returns true for collision
        """
        agent = self.auto_uavs_dict[agent_id]
        collision_with_stat_obj, _ = agent.get_state_static_obj(
            self.airspace.location_utm_hospital.geometry, "collision"
        )
        other_agent_list = agent.get_other_uav_list(self.auto_uavs_list)
        collision_with_dyn_obj = agent.get_collision(other_agent_list)

        collision_detected = collision_with_dyn_obj or collision_with_stat_obj
        if collision_detected:
            truncated = True
        else:
            truncated = False

        return truncated

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

        reward_sum *= self.current_time_step

        return reward_sum

    def get_vertiport_from_atc(self) -> None:
        """This is a convinience method, for reset()"""

        vertiports_point_array = [
            vertiport.location for vertiport in self.atc.vertiports_in_airspace
        ]
        self.sim_vertiports_point_array = vertiports_point_array

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
        self.current_time_step = 0
        self.auto_uavs_list = []
        self.atc.auto_uavs_list = []
        self.atc.vertiports_in_airspace = []
        self.sim_vertiports_point_array = []

        self.atc.create_n_random_vertiports(self.num_vertiports)
        self.atc.create_n_auto_uavs(self.num_auto_uav)
        self.get_vertiport_from_atc()

        self.auto_uavs_list = self.atc.auto_uavs_list
        self.auto_uavs_dict = {
            auto_uav.id: auto_uav for auto_uav in self.auto_uavs_list
        }
        self.possible_agents = list(self.auto_uavs_dict.keys())

        # agents should be a list of agent_id = auto_uav.id
        self.agents = self.possible_agents
        # self.num_agents = len(self.agents)

        self.rewards = {agent: 0 for agent in self.agents}
        # self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        # self.state = {agent: NONE for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}

        return self.observations, self.infos

    def step(self, actions: dict) -> tuple[dict, dict, dict, dict, dict]:
        """
        Step function advances the entire simulation

        Args:
            Actions (dict): The action the UAV takes

        Return:
            observations (dict): Feeds the state space of the agent
            rewards (int): The reward earned durring the step
            terminations (bool): Checks if the agent has reached its destination
            truncations (bool): Checks for collisions with static and dynamic objects
            infos (dict): distance of agent from target vertiport
        """
        for agent_id in actions:
            action = actions[agent_id]
            self.auto_uavs_dict[agent_id].step(action[0], action[1])

            obs = self._get_obs(agent_id)
            reward = self.get_reward(obs)
            termination = self.has_terminated(agent_id)
            truncation = self.has_truncated(agent_id)
            info = self._get_info(agent_id)

            #       dict[key]     = value -------  value type
            self.observations[agent_id] = obs  #
            self.rewards[agent_id] = reward
            self.terminations[agent_id] = termination
            self.truncations[agent_id] = truncation
            self.infos[agent_id] = info

        # Every time step is called the current time step increments by one second
        self.current_time_step += 1

        return (
            self.observations,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def render_init(
        self,
    ) -> None:
        """
        Initalizes the rendering

        Returns:
            fig(plt.Figure): The outside of the graph that is rendered
            ax(plt.Axes): The backdrop of the graph
        """
        fig, ax = plt.subplots()
        return fig, ax

    def render_static_asset(self, ax: plt.Axes):
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

    def render(self, fig: Figure, ax: Axes):
        """
        Renders everything in the graph

        Args:
            fig(plt.Figure): The outside of the graph that is rendered
            ax(plt.Axes): The backdrop of the graph
        """
        plt.cla()
        self.render_static_asset(ax)

        for auto_uav in self.auto_uavs_list:
            auto_uav_footprint_poly = auto_uav.uav_polygon_plot(
                auto_uav.collision_radius
            )
            auto_uav_footprint_poly.plot(
                ax=ax, color=auto_uav.uav_footprint_color, alpha=0.3
            )

            auto_uav_nmac_poly = auto_uav.uav_polygon_plot(auto_uav.nmac_radius)
            auto_uav_nmac_poly.plot(
                ax=ax, color=auto_uav.uav_nmac_radius_color, alpha=0.3
            )

            auto_uav_detection_poly = auto_uav.uav_polygon_plot(
                auto_uav.detection_radius
            )
            auto_uav_detection_poly.plot(
                ax=ax, color=auto_uav.uav_detection_radius_color, alpha=0.3
            )
            auto_x_current, auto_y_current, auto_dx_current, auto_dy_current = (
                auto_uav.get_uav_current_heading_arrow()
            )
            ax.arrow(
                auto_x_current,
                auto_y_current,
                auto_dx_current,
                auto_dy_current,
                alpha=1,
            )
            auto_x_final, auto_y_final, auto_dx_final, auto_dy_final = (
                auto_uav.get_uav_final_heading_arrow()
            )
            ax.arrow(
                auto_x_final, auto_y_final, auto_dx_final, auto_dy_final, alpha=0.8
            )

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(self.sleep_time)

    def get_agent_speed(
        self, auto_uav: AutonomousUAV
    ) -> np.ndarray:  # TODO #8 - rename to get_agent_speed()
        return np.array([auto_uav.current_speed])

    def get_agent_deviation(self, auto_uav: AutonomousUAV) -> np.ndarray:
        #! should this be converted between -180 to 180
        return np.array(
            [auto_uav.current_heading_deg - auto_uav.current_ref_final_heading_deg]
        )

    def _get_obs(self, agent_id: str) -> dict:
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
        agent = self.auto_uavs_dict[agent_id]

        # need a new method for this
        agent_speed = self.get_agent_speed(agent)
        # need a new method for this
        agent_deviation = self.get_agent_deviation(agent)

        other_auto_uav = agent.get_other_uav_list(self.auto_uavs_list)
        # change this list -------------------------------->needs to be the auto_uav_list
        intruder_info = agent.get_state_dynamic_obj(
            other_auto_uav, "nmac"
        )  # self.nmac_with_dynamic_obj()

        # TODO - need to change uav to auto_uav
        if intruder_info:
            intruder_detected = 1
            intruder_id = np.array([intruder_info["intruder_id"]])
            intruder_pos = intruder_info["intruder_pos"]
            intruder_heading = np.array([intruder_info["intruder_current_heading"]])
            distance_to_intruder = np.array(
                [agent.current_position.distance(intruder_pos)]
            )
            relative_heading_intruder = np.array(
                [agent.current_heading_deg - float(intruder_heading)]
            )
        else:
            intruder_detected = 0
            intruder_id = np.array([0])
            distance_to_intruder = np.array([0])
            relative_heading_intruder = np.array([0])
            intruder_heading = np.array([0])

        #!restricted airspace
        #!implementatation will require updating observation space in __init__
        restricted_airspace, _ = agent.get_state_static_obj(
            self.atc.airspace.location_utm_hospital.geometry,
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

    def _get_info(self, agent_id: str) -> dict:
        """
        Gets the distance between the uav and target vertiport

        Args:
            agent_id (str): The ID of the target UAV

        Returns:
            (dict): distance to target vertiport
        """
        auto_uav = self.auto_uavs_dict[agent_id]

        return {
            "distance_to_end_vertiport": auto_uav.current_position.distance(
                auto_uav.end_point
            )
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent_id: str) -> spaces.Dict:
        """
        Creates the observation space for the agents

        Args:
            agent_id (str): The ID of the target UAV

        Returns:
            (spaces.Dict): state space of the agent
                agent_id (int32): The identification number of the target UAV
                agent_speed (float64): The speed of the agent UAV
                agent_deveation (float64): The angular deveation from the path to the target vertiport
                intruder_detected (spaces.Discrete): Boolean that returns true when an intruder is detected
                intruder_id (int32): The identification number of the intruding UAV
                distance_to_intruder (float64): The distance to the UAV in meters
                relative_heading_intruder (float64): The relative heaving to the intruder
                intruder_current_heading (float64): The heading at wich the intruder is traveling
        """
        return spaces.Dict(
            {
                "agent_id": spaces.Box(
                    low=0, high=np.iinfo(np.int32).max, shape=(1,), dtype=np.int32
                ),
                "agent_speed": spaces.Box(
                    low=0,
                    high=self.auto_uav_max_speed,
                    shape=(1,),
                    dtype=np.float64,
                ),
                "agent_deviation": spaces.Box(
                    low=-180, high=180, shape=(1,), dtype=np.float64
                ),
                "intruder_detected": spaces.Discrete(2),
                "intruder_id": spaces.Box(
                    low=0, high=np.iinfo(np.int32).max, shape=(1,), dtype=np.int32
                ),
                "distance_to_intruder": spaces.Box(
                    low=0,
                    high=self.auto_uav_detection_radius,
                    shape=(1,),
                    dtype=np.float64,
                ),
                "relative_heading_intruder": spaces.Box(
                    low=-180, high=180, shape=(1,), dtype=np.float64
                ),
                "intruder_current_heading": spaces.Box(
                    low=-180, high=180, shape=(1,), dtype=np.float64
                ),
            }
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent_id: str) -> spaces.Box:
        return spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64)
