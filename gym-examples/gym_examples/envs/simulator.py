# Welcome to the simulator, this takes all the classes from all the modules, and builds an instance of the simulator
from airspace import Airspace
from airtrafficcontroller import ATC
from uav import UAV
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
import geopandas as gpd
from shapely import Point
import time
from typing import List, Callable

"""
    When we create the UAM env(subclass of gymEnv) it will build an instance that is similar to simulator.
    The initializer arguments of UAM_Env will be passed to the simulator, that is location_name, basic_uav_no, vertiport_no, and Auto_uav(only one for now)
    [** emphasizing, the above arguments are arguments of UAV_Env passed to simulator_env**]
    
    Inside the simulator there will be one instance of Auto_UAV, this Auto_UAV's argument is a tuple of actions defined in UAV_Env.
    The Auto_UAV navigates the airspace using these actions. 

    *** The "step" method of UAV_Env, is used to step every uav(meaning basic_uav and Auto_uav)

    ***Refer to uam_single_agent_env's TRAINING section for questions that need to be answered, for further documentation and clarification

     
"""


class Simulator:

    def __init__(
        self,
        location_name: str,
        num_vertiports: int,
        num_basic_uavs: int,
        sleep_time: float,
        total_timestep: int,
    ) -> None:
        """
        Initializes a Simulator object.

        Args:
            location_name (str): The name of the location for the simulation.
            num_vertiports (int): The number of vertiports to create in the simulation.
            num_basic_uavs (int): The number of basic UAVs to create in the simulation.
            sleep_time (float): Time to sleep after each step
            total_timestep (int): Total number of timesteps in simulation
        """
        # sim airspace and ATC
        self.airspace = Airspace(location_name=location_name)
        self.atc = ATC(
            self.airspace,
        )
        # Initialize sim's vertiports and uavs using ATC
        # *
        #! These two statements will most probably be in reset()
        #! Think how a seed value can be added to the system, so that the system is reproduceable - the vertiports and the uav assignment fixed to some seed value.
        self.atc.create_n_random_vertiports(num_vertiports)
        self.atc.create_n_basic_uavs(
            num_basic_uavs,
        )

        # unpacking atc.vertiports in airspace
        vertiports_point_array = [
            vertiport.location for vertiport in self.atc.vertiports_in_airspace
        ]
        # sim data
        self.sim_vertiports_point_array = vertiports_point_array
        self.uav_list: List[UAV] = self.atc.basic_uav_list
        # *
        # sim sleep time
        self.sleep_time = sleep_time
        # sim run time
        self.total_timestep = total_timestep

    def render(
        self,
        fig: Figure,
        ax: Axes,
        static_plot: Callable,
        sim: "Simulator",
        gpd: gpd.GeoDataFrame,
    ) -> None:
        """
        Renders everything in the graph

        Args:
            fig(plt.Figure): The outside of the graph that is rendered
            ax(plt.Axes): The backdrop of the graph
            static_plot (Callable): Function that plots static objects
            sim (Simulator):
            gdp (gdp.GeoDataFrame):
        """
        plt.cla()
        static_plot(sim, ax, gpd)
        # UAV PLOT LOGIC
        for uav_obj in self.uav_list:
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

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(self.sleep_time)

    def sim_step(self, action_list: list) -> list[tuple[str, Point]]:
        """
        Advances the simulation one time step

        Args:
            action_list (list): A list of actions for the UAVs in the simulation
        """
        obs_list = []
        for uav_id, action in action_list:
            # print(uav_id, action)
            uav = self.get_uav(uav_id)
            self.atc.has_left_start_vertiport(uav)
            self.atc.has_reached_end_vertiport(uav)
            obs = uav.step(action)
            assert uav.id == uav_id
            obs_list.append((uav_id, obs))
        return obs_list

    #! arg -> action_list needs to be unpacked, action_list should contain (action, uav_id) tuple,
    #! which will be unpacked and assigned to each uav based on uav_id
    def get_action_list(self, controller_predict: Callable) -> list[tuple[str, float]]:
        action_list = []
        for uav in self.uav_list:
            #                    state -> ((bool,                float                  ), dict            | None)
            #                               static_obj_detected, uav.current_heading_deg , dynamic_obj_info| None
            obs = self._get_obs(uav)
            action = controller_predict(obs)
            action_list.append((uav.id, action))
        return action_list

    def get_uav(self, uav_id: str) -> UAV:
        """Returns the UAV properties when given a uav_id"""
        for uav in self.uav_list:
            if uav.id == int(uav_id):
                return uav
            else:
                continue
        raise RuntimeError("UAV not it list")

    def run_simulator(
        self,
        fig: Figure,
        ax: Axes,
        static_plot: Callable,
        sim: "Simulator",
        gpd: gpd.GeoDataFrame,
        controller_predict: Callable,
    ) -> (
        None
    ):  #! das-controller needs to be changed to controller_predict implement uniform name all across code base
        """
        Runs the simulator.
        This method packs rendering, and stepping into one method.
        Generally, for RL the loop is written explictly. This method was written for convinience.

        Args:
            fig (matplotlib.figure.Figure): The figure object for plotting.
            ax (matplotlib.axes.Axes): The axes object for plotting.
            static_plot (function): A function that plots the static elements of the simulation.

        Returns:
            None
        """
        for _ in range(self.total_timestep):
            self.render(fig, ax, static_plot, sim, gpd)
            #! need an initial state
            # - initial state is created by create_n_basic_uavs
            #! feed the initial state to the controller if controller
            action_list = self.get_action_list(
                controller_predict
            )  #! check for zero_controller
            self.sim_step(action_list)  #! how would step behave to action_list

        print("Simulation complete.")

    def _get_obs(self, uav_obj: UAV) -> tuple[tuple[bool, float], None | dict]:
        state_info = uav_obj.get_state(
            self.uav_list, self.airspace.location_utm_hospital_buffer
        )

        return state_info

    # #TODO - this is necessary
    # def reset(self,):
    #     pass

    # #TODO - this is necessary
    # def close(self,):
    #     pass

    # def sim_step(self, action_list):
    #     obs_list = []

    #     # UAV VERTIPORT REASSIGNMENT LOGIC
    #     for uav_obj in self.atc.basic_uav_list:
    #         self.atc.has_left_start_vertiport(uav_obj)
    #         self.atc.has_reached_end_vertiport(uav_obj)

    #     # UAV STEP LOGIC
    #     for uav_obj in self.uav_list:
    #         #! get_obs() -> state_info, controller should accept state_info, das_controller(state_info)
    #         #intruder_state_info = uav_obj.get_state(self.uav_list)
    #         intruder_state_info = self._get_obs(uav_obj)
    #         #! step should accept action, step(action)
    #         if das_controller is None:
    #             action = None
    #         else:
    #             action = das_controller.get_action(intruder_state_info)
    #     #! sim_step has to return observation
    #     #! but sim_step, steps all basic_uavs in the airspace,
    #     #! so the obs will have to be a list that contains all the obs information about all uavs
    #         obs = uav_obj.step(action)
    #         obs_list.append((uav_obj.id,obs))

    #     return obs_list
