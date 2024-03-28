#Welcome to the simulator, this takes all the classes from all the modules, and builds an instance of the simulator
from airspace import Airspace
from airtrafficcontroller import ATC
from uav import UAV
import matplotlib.pyplot as plt
import geopandas as gpd
import time
from typing import List

'''
    Read before continuing. 
    
    When we create the UAM env(subclass of gymEnv) it will build an instance of the simulator.
    The initializer arguments of UAM_Env will be passed to the simulator, that is location_name, reg_uav_no, vertiport_no, and Auto_uav(only one for now)
    [** emphasizing, the above arguments are arguments of UAV_Env passed to simulator_env**]
    
    Inside the simulator there will be one instance of Auto_UAV, this Auto_UAV's argument is a tuple of actions defined in UAV_Env.
    The Auto_UAV navigates the airspace using these actions. 

    *** The "step" method of UAV_Env, is used to step every uav(meaning reg_uav and Auto_uav)

    ***Refer to uam_single_agent_env's TRAINING section for questions that need to be answered, for further documentation and clarification

     
'''

class Simulator:
    def __init__(self, location_name, num_vertiports, num_reg_uavs, sleep_time, total_timestep): 
        """
        Initializes a Simulator object.

        Args:
            location_name (str): The name of the location for the simulation.
            num_vertiports (int): The number of vertiports to create in the simulation.
            num_reg_uavs (int): The number of regular UAVs to create in the simulation.
        """       
        # sim airspace and ATC
        self.airspace = Airspace(location_name=location_name)
        self.atc = ATC(self.airspace)
        # Initialize sim's vertiports and uavs using ATC 
        self.atc.create_n_random_vertiports(num_vertiports)
        self.atc.create_n_reg_uavs(num_reg_uavs,)
        # unpacking atc.vertiports in airspace
        vertiports_point_array = [vertiport.location for vertiport in self.atc.vertiports_in_airspace]
        # sim data
        self.sim_vertiports_point_array = vertiports_point_array
        self.uav_list:List[UAV] = self.atc.reg_uav_list 
        # sim sleep time
        self.sleep_time = sleep_time
        #sim run time
        self.total_timestep = total_timestep

    
    
    def RUN_SIMULATOR(self, fig, ax, static_plot, sim, gpd):
        """
        Runs the simulator.

        Args:
            fig (matplotlib.figure.Figure): The figure object for plotting.
            ax (matplotlib.axes.Axes): The axes object for plotting.
            static_plot (function): A function that plots the static elements of the simulation.

        Returns:
            None
        """
        for _ in range(self.total_timestep):
            plt.cla()
            static_plot(sim, ax, gpd)

            # PLOT LOGIC
            for uav_obj in self.uav_list:
                gpd.GeoSeries(uav_obj.current_position).plot(ax=ax, color='red', alpha=0.3)
                #TODO - wrap uav with Point instead of gpd.Geoseries
                gpd.GeoSeries(uav_obj.current_position).buffer(60).plot(ax=ax, color='yellow', alpha=0.2)
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(self.sleep_time)

            # UAV VERTIPORT REASSIGNMENT LOGIC
            for uav_obj in self.atc.reg_uav_list:
                self.atc.has_left_start_vertiport(uav_obj)
                self.atc.has_reached_end_vertiport(uav_obj)

            # UAV STEP LOGIC
            for uav_obj in self.uav_list: #! all uavs are stepping
                uav_obj.step()

            #TODO - collision detection and avoidance logic
            

            

        print('Simulation complete.')

    

    

