#Welcome to the simulator, this takes all the classes from all the modules, and builds an instance of the simulator
import airspace, airtrafficcontroller, uav, autonomous_uav, vertiport
import matplotlib.pyplot as plt
import geopandas as gpd
import time
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
    def __init__(self, location_name, num_vertiports, num_reg_uavs):
        self.airspace = airspace.Airspace(location_name=location_name)
        self.atc = airtrafficcontroller.ATC(self.airspace)
        self.vertiports = self.atc.create_n_random_vertiports(num_vertiports)
        self.atc.create_n_reg_uavs(num_reg_uavs, self.vertiports)
        self.uav_list = self.atc.reg_uav_list

    
    
    
    def RUN_SIMULATOR(self, fig, ax, static_plot,):
        while True:
            plt.cla()
            static_plot()
            for uav_obj in self.uav_list: #! all uavs are stepping
                uav_obj.step() 
            for uav_obj in self.uav_list:
                gpd.GeoSeries(uav_obj.current_position).plot(ax=ax, color='red', alpha=0.3)
                gpd.GeoSeries(uav_obj.current_position).buffer(60).plot(ax=ax, color='yellow', alpha=0.2)
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.1)
            
            for uav_obj in self.uav_list:
                if uav_obj.current_position.distance(uav_obj.end_point) <= uav_obj.proximity:
                    uav_obj.reached = True
            
            all_reached = all([uav_obj.reached for uav_obj in self.uav_list])
            if all_reached:
                break



        print('Simulation complete. Path traces of UAVs are ready')
    

    

