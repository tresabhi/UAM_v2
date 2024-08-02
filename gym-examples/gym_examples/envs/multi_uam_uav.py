import functools
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from matplotlib.animation import FuncAnimation
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
        

        # Animation data
        data = {'current_time_step':[],
                'auto_uav_id':[],
                'auto_uav':[],
                'current_position':[],
                'current_heading':[],
                'final_heading':[]}
        self.df = pd.DataFrame(data)
        
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

    def add_data(self,auto_uav:AutonomousUAV):
        self.df = self.df._append({
            'current_time_step':self.current_time_step,
            'auto_uav_id':auto_uav.id,
            'auto_uav':auto_uav,
            'current_position':auto_uav.current_position,
            'current_heading':auto_uav.current_heading_radians,
            'final_heading':auto_uav.current_ref_final_heading_rad},
            ignore_index = True)
    
    
    
    
    
    def has_terminated(self, agent_id: str) -> bool:
        agent = self.auto_uavs_dict[agent_id]
        dist_to_end_point = agent.current_position.distance(agent.end_point)

        if dist_to_end_point < agent.landing_proximity:
            terminated = True
        else:
            terminated = False

        return terminated

    def has_truncated(self, agent_id: str) -> bool:
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

        reward_to_destination = float(obs["agent_velocity"]) * float(
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


        # List of all AUTO UAV  for animation 
        #! might be redundant 
        self.auto_uav_in_airspace = self.atc.auto_uavs_list
        
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

        for agent_id in actions:
            # Adding data to data_frame for animation 
            self.add_data(self.auto_uavs_dict[agent_id])
            
            # Step method 
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
        fig, ax = plt.subplots()
        return fig, ax

    def render_static_asset(
        self, ax: plt.Axes
    ):
        self.airspace.location_utm_gdf.plot(ax=ax, color="gray", linewidth=0.6)
        self.airspace.location_utm_hospital_buffer.plot(ax=ax, color="red", alpha=0.3)
        self.airspace.location_utm_hospital.plot(ax=ax, color="black")
        # adding vertiports to static plot
        gpd.GeoSeries(self.sim_vertiports_point_array).plot(ax=ax, color="black")

    def render(self, fig: Figure, ax: Axes):
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
            auto_x_current, auto_y_current, auto_dx_current, auto_dy_current = auto_uav.get_uav_current_heading_arrow()
            ax.arrow(auto_x_current, auto_y_current, auto_dx_current, auto_dy_current, alpha=1)
            auto_x_final, auto_y_final, auto_dx_final, auto_dy_final = auto_uav.get_uav_final_heading_arrow()
            ax.arrow(auto_x_final, auto_y_final, auto_dx_final, auto_dy_final, alpha=0.8)

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(self.sleep_time)

    def get_agent_velocity(
        self, auto_uav: AutonomousUAV
    ) -> np.ndarray:  # TODO #8 - rename to get_agent_speed()
        return np.array([auto_uav.current_speed])

    def get_agent_deviation(self, auto_uav: AutonomousUAV) -> np.ndarray:
        #! should this be converted between -180 to 180
        return np.array(
            [auto_uav.current_heading_deg - auto_uav.current_ref_final_heading_deg]
        )

    def _get_obs(self, agent_id: str) -> dict:
        agent = self.auto_uavs_dict[agent_id]

        # need a new method for this
        agent_velocity = self.get_agent_velocity(agent)
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
            "agent_velocity": agent_velocity,
            "agent_deviation": agent_deviation,
            "intruder_detected": intruder_detected,
            "intruder_id": intruder_id,
            "distance_to_intruder": distance_to_intruder,
            "relative_heading_intruder": relative_heading_intruder,
            "intruder_current_heading": intruder_heading,
        }

        return observation

    def _get_info(self, agent_id: str) -> dict:

        auto_uav = self.auto_uavs_dict[agent_id]

        return {
            "distance_to_end_vertiport": auto_uav.current_position.distance(
                auto_uav.end_point
            )
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent_id: str) -> spaces.Dict:
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


    
    def get_data_at_timestep(self,timestep):
        filtered_df = self.df[self.df['current_time_step']== timestep]
        return filtered_df[['auto_uav_id', 'auto_uav', 'current_position', 'current_heading', 'final_heading']]
    

    def get_animate_fig_ax(self):
        fig, ax = plt.subplots()
        return fig, ax

    def init_animate(self, animate_ax):
        self.render_static_asset(animate_ax)
        return []
    
    def update_animate(self, frame, animate_ax):
        plt.cla()
        self.render_static_assets(animate_ax)
        data_frame = self.get_data_at_timestep(frame)
        artists = []

        for i, row in data_frame.iterrows():
            if isinstance(row['uav'], AutonomousUAV):
                autouav_obj:AutonomousUAV = row['uav']
                autouav = gpd.GeoSeries([row['current_position']])
                current_heading = row['current_heading']
                final_heading = row['final_heading']
                auto_uav_detection = autouav.buffer(autouav_obj.detection_radius).plot(color = autouav_obj.uav_detection_radius_color,ax=animate_ax)
                auto_uav_nmac = autouav.buffer(autouav_obj.nmac_radius).plot(color=autouav_obj.uav_nmac_radius_color,ax=animate_ax)
                
                x,y = row['current_position'].x, row['current_position'].y 
                r = autouav_obj.detection_radius
                dx, dy = r*np.cos(current_heading), r*np.sin(current_heading)
                auto_uav_current_heading_arrow = animate_ax.arrow(x,y,dx,dy, alpha=1)
                artists.append(auto_uav_detection)
                artists.append(auto_uav_nmac)
                artists.append(auto_uav_current_heading_arrow)
            else:
                raise RuntimeError('Update animate has instance type error')
        return artists
    
    def create_animation(self, env_time_step):
        fig, ax = self.get_animate_fig_ax()
        df = self.df
        ani = FuncAnimation(fig, self.update_animate, frames=range(0,env_time_step), fargs=[ax])
        return ani
    
    def save_animation(self, animation_obj, file_name):
        animation_obj.save(file_name+'.mp4', writer='ffmpeg')
    