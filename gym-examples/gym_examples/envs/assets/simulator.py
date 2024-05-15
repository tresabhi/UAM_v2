#Welcome to the simulator, this takes all the classes from all the modules, and builds an instance of the simulator
from airspace import Airspace
from airtrafficcontroller import ATC
from uav import UAV
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely import Point
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

'''
since the simulator instance will follow gym_env remove total_timestep, 
in __main__ create an instance of the simulator(env), and run that for total_timesteps 
'''

class Simulator:
    #TODO - remove total_timestep and 
    #TODO - sleep_time should be defined inside the class 
    def __init__(self, location_name, num_vertiports, num_reg_uavs, sleep_time, total_timestep, controller): 
        """
        Initializes a Simulator object.

        Args:
            location_name (str): The name of the location for the simulation.
            num_vertiports (int): The number of vertiports to create in the simulation.
            num_reg_uavs (int): The number of regular UAVs to create in the simulation.
        """       
        # sim airspace and ATC
        self.airspace = Airspace(location_name=location_name)
        self.atc = ATC(self.airspace, controller)
        # uav controller 
        #self.controller = controller
        # Initialize sim's vertiports and uavs using ATC 
        self.atc.create_n_random_vertiports(num_vertiports)
        self.atc.create_n_reg_uavs(num_reg_uavs,controller)
        # unpacking atc.vertiports in airspace
        vertiports_point_array = [vertiport.location for vertiport in self.atc.vertiports_in_airspace]
        # sim data
        self.sim_vertiports_point_array = vertiports_point_array
        self.uav_list:List[UAV] = self.atc.reg_uav_list 
        # sim sleep time
        self.sleep_time = sleep_time
        #sim run time
        self.total_timestep = total_timestep

    
    #TODO - break this method so that all the subfunctions are distributed to the methods below
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
            #TODO - implement the env step function - we could rename RUN_SIMULATOR and organise the input arguments
            #* From here 
            plt.cla()
            static_plot(sim, ax, gpd)

            # UAV PLOT LOGIC
            for uav_obj in self.uav_list:
                #gpd.GeoSeries(uav_obj.current_position).plot(ax=ax, color='red', alpha=0.3)
                uav_footprint_poly = uav_obj.uav_footprint_polygon()
                uav_nmac_poly = uav_obj.uav_nmac_polygon()
                uav_footprint_poly.plot(ax=ax, color=uav_obj.uav_footprint_color, alpha=0.3)
                uav_nmac_poly.plot(ax=ax, color=uav_obj.uav_nmac_radius_color, alpha=0.3)
                #TODO - wrap uav with Point instead of gpd.Geoseries
                #gpd.GeoSeries(uav_obj.current_position).buffer(800).plot(ax=ax, color='yellow', alpha=0.2)
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(self.sleep_time)

            # UAV VERTIPORT REASSIGNMENT LOGIC
            for uav_obj in self.atc.reg_uav_list:
                self.atc.has_left_start_vertiport(uav_obj)
                self.atc.has_reached_end_vertiport(uav_obj)

            # UAV STEP LOGIC
            for uav_obj in self.uav_list: #! all uavs are stepping
                uav_obj.step(self.airspace.location_utm_hospital, self.airspace.location_utm_hospital_buffer, self.uav_list)

            #TODO - remove this section of code, integrate the collision inside UAVs step function

            # Collision detection and avoidance logic
            # for uav_obj in self.uav_list:
            #     uav_obj.static_nmac_detection(self.airspace.location_utm_hospital_buffer)
            #     uav_obj.static_collision_detection(self.airspace.location_utm_hospital)
            #     uav_obj.uav_collision_detection(self.uav_list)
            #     uav_obj.uav_nmac_detection(self.uav_list)
            #* upto here - all of this will be wrapped in a env step function 

            

        print('Simulation complete.')

    def __init__(self, airspace,airtrafficcontroller,uav,autonomous_uav,vertiport,render_mode=None):
        pass 

    def _get_obs(self,):
        pass 

    def _get_info(self,):
        pass 

    #TODO - this is necessary 
    def reset(self,):
        pass 
    
    #TODO - this is necessary 
    def step(self,action):
        pass
    
    #TODO - this is necessary 
    def render(self,):
        pass

    def _render_frame(self,):
        pass

    #TODO - this is necessary 
    def close(self,):
        pass

    

    

