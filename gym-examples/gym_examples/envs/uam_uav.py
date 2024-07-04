import numpy as np 
import matplotlib.pyplot as plt
from typing import List
import time
from geopandas import GeoSeries
import geopandas as gpd
import gymnasium as gym
from gymnasium import spaces

from airspace import Airspace 
from airtrafficcontroller import ATC
from uav_basic import UAV_Basic
from autonomous_uav import Autonomous_UAV
from vertiport import Vertiport





'''
Remember this environment will only be used for training,
Therefore there will only be one auto_uav which is built into the uam_env 
'''

'''
    When we create the UAM env(subclass of gymEnv) it will build an instance that is similar to simulator.
    The initializer arguments of UAM_Env will be passed to the simulator, that is location_name, reg_uav_no, vertiport_no, and Auto_uav(only one for now)
    [** emphasizing, the above arguments are arguments of UAV_Env passed to simulator_env**]
    
    Inside the simulator there will be one instance of Auto_UAV, this Auto_UAV's argument is a tuple of actions defined in UAV_Env.
    The Auto_UAV navigates the airspace using these actions. 

    *** The "step" method of UAV_Env, is used to step every uav_basic(meaning reg_uav and Auto_uav)

    ***Refer to uam_single_agent_env's TRAINING section for questions that need to be answered, for further documentation and clarification

     
'''



class Uam_Uav_Env(gym.Env):
    metadata = {"render_mode":["human", "rgb_array"], "render_fps":4}

    def __init__(self, location_name, num_vertiport, num_reg_uav, sleep_time=0.005, render_mode=None):
        # Environment attributes
        self.current_time_step = 0
        self.num_vertiports = num_vertiport
        self.num_reg_uavs = num_reg_uav
        self.sleep_time = sleep_time
        self.airspace = Airspace(location_name)
        self.atc = ATC(airspace=self.airspace)
        
        # Vertiport initialization
        self.atc.create_n_random_vertiports(num_vertiport)
        
        # UAV initialization
        self.atc.create_n_reg_uavs(num_reg_uav)
        
        # Environment data
        vertiports_point_array = [vertiport.location for vertiport in self.atc.vertiports_in_airspace]
        self.sim_vertiports_point_array = vertiports_point_array 
        self.uav_basic_list: List[UAV_Basic] = self.atc.reg_uav_list
        
        # Auto UAV initialization
        start_vertiport_auto_uav = self.get_start_vertiport_auto_uav()
        end_vertiport_auto_uav = self.get_end_vertiport_auto_uav(start_vertiport_auto_uav)
        self.auto_uav = Autonomous_UAV(start_vertiport_auto_uav, end_vertiport_auto_uav)
 
        # Environment spaces 
        self.observation_space = spaces.Dict({
            'agent_id': spaces.Box(low=0, high=np.iinfo(np.int64).max, shape=(1,), dtype=np.int64),  # Agent ID as integer
            'agent_velocity': spaces.Box(low=-self.auto_uav.max_speed, high=self.auto_uav.max_speed, shape=(1,), dtype=np.float64),  # agent's velocity #! need to check why this is negative 
            'agent_deviation': spaces.Box(low=-360, high=360, shape=(1,), dtype=np.float64),  # agent's heading deviation #!this needs to be corrected to -180 to 180
            'intruder_detected': spaces.Discrete(2),  # 0 for no intruder, 1 for intruder detected
            'intruder_id': spaces.Box(low=0, high=np.iinfo(np.int64).max, shape=(1,), dtype=np.int64),  # Intruder ID as integer
            'distance_to_intruder': spaces.Box(low=0, high=self.auto_uav.detection_radius, shape=(1,), dtype=np.float64),  # Distance to intruder
            'relative_heading_intruder': spaces.Box(low=-360, high=360, shape=(1,), dtype=np.float64),  # Relative heading of intruder #!this needs to be corrected to -180 to 180
            'intruder_current_heading': spaces.Box(low=-180, high=180, shape=(1,), dtype=np.float64)  # Intruder's heading
        })

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64)  # Normalized action space


        


    def get_vertiport_from_atc(self):
        '''This is a convinience method, for reset()'''
        vertiports_point_array = [vertiport.location for vertiport in self.atc.vertiports_in_airspace]
        self.sim_vertiports_point_array = vertiports_point_array

    def get_uav_list_from_atc(self):
        '''This is a convinience method, for reset()'''
        self.uav_basic_list = self.atc.reg_uav_list


    def reset(self,seed=None, options=None):
        '''
        This method will reset the environment. NOT sure how this will be done, but it will be done.
        '''
        # this will first remove the existing vertiports 
        #                 remove the existing reg uavs
        #                 remove the auto_uav
        # then it will create new vertiport 
        #              create new reg_uavs
        # then this method will use one of the empty vertiports and assign it to the auto_uav
        
        '''
        Since reset is called after creating the env, in reset we will perform the following 
        
        1) create the auto_uav with start_vertiport -> empty_vertiport, end_vertiport -> choose a random vertiport from vertiport list this function will be performed by the ATC
        '''
        #! check the following things - 1) how and where is a uav_basic created, 
        #! how are its attributes assigned, 
        #! once this is clear -> create a script for auto_uav, and use that to create the auto_uav above in reset and assign all the necessary attributes to it

        super().reset(seed=seed)
        self.np_random, seed = gym.utils.seeding.np_random(seed)

        self.current_time_step = 0
        self.atc.reg_uav_list = []
        self.atc.vertiports_in_airspace = []
        self.uav_basic_list = []
        self.sim_vertiports_point_array = []

        self.atc.create_n_random_vertiports(self.num_vertiports)
        self.atc.create_n_reg_uavs(self.num_reg_uavs)
        self.get_vertiport_from_atc()
        self.get_uav_list_from_atc()
        #! Need to perform similar actions for auto_uav as well -> like going through the list of vertiports and find the empty one and assigning it to the auto_uav
        self.auto_uav = None
        start_vertiport_auto_uav = self.get_start_vertiport_auto_uav() # go through all the vertiports 
        end_vertiport_auto_uav = self.get_end_vertiport_auto_uav(start_vertiport_auto_uav)
        self.auto_uav = Autonomous_UAV(start_vertiport_auto_uav, end_vertiport_auto_uav) 


        #! !! reset() needs to return observation !!
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def get_agent_velocity(self,):
        return np.array([self.auto_uav.current_speed])
    
    def get_agent_deviation(self,):
        #! might need to convert the current degree between -180 to 180
        return np.array([self.auto_uav.current_heading_deg - self.auto_uav.current_ref_final_heading_deg])
    
    
    def _get_obs(self):
        agent_id = np.array([self.auto_uav.id])
        agent_velocity = self.get_agent_velocity()  
        agent_deviation = self.get_agent_deviation()  
        intruder_info = self.nmac_with_dynamic_obj()
        
        if intruder_info:
            intruder_detected = 1
            intruder_id = np.array([intruder_info['intruder_id']])
            intruder_pos = intruder_info['intruder_pos']
            intruder_heading = np.array([intruder_info['intruder_current_heading']])
            distance_to_intruder = np.array([self.auto_uav.current_position.distance(intruder_pos)])
            relative_heading_intruder = np.array([self.auto_uav.current_heading_deg - float(intruder_heading)])
        else:
            intruder_detected = 0
            intruder_id = np.array([0])
            distance_to_intruder = np.array([0])
            relative_heading_intruder = np.array([0])
            intruder_heading = np.array([0])
            
        observation = {
            'agent_id': agent_id,
            'agent_velocity': agent_velocity,
            'agent_deviation': agent_deviation,
            'intruder_detected': intruder_detected,
            'intruder_id': intruder_id,
            'distance_to_intruder': distance_to_intruder,
            'relative_heading_intruder': relative_heading_intruder,
            'intruder_current_heading': intruder_heading
        }
        
        return observation


         

    def _get_info(self,):

        return {'distance_to_end_vertiport':self.auto_uav.current_position.distance(self.auto_uav.end_point)}

    def set_uav_intruder_list(self):
        for uav in self.uav_basic_list:
            uav.get_intruder_uav_list(self.uav_basic_list)
    
    def set_building_gdf(self):
        for uav in self.uav_basic_list:
            uav.get_airspace_building_list(self.airspace.location_utm_hospital_buffer)

    # def set_auto_uav_intruder_list(self):
    #     self.auto_uav.get_intruder_uav_list(self.uav_basic_list)
    
    def set_auto_uav_building_gdf(self):
        #! might need to set building property for auto_uav
        # self.auto_uav.get_airspace_building_list(self.airspace.location_utm_hospital_buffer)
        self.auto_uav.get

    def step(self,action):
        '''
        This method is used to step the environment, it will step the environment by one timestep.
        
        The action argument - will be passed to auto_uav's step method
        
        Regular UAVs will step without action. so I will need to modify regular uav_basic in such a way that they will step without action. 
        This tells me that regular uav_basic will need to have collision avoidance built into the uav_basic module, such that they can step without action. 
         
        '''
        #decomposing action tuple 
        acceleration = action[0]
        heading_correction = action[1]
        
        self.set_uav_intruder_list()
        self.set_building_gdf()
        
        
        
        #for uav_basic in uav_basic_list step all uav_basic
        for uav_basic in self.uav_basic_list:
            self.atc.has_left_start_vertiport(uav_basic)
            self.atc.has_reached_end_vertiport(uav_basic)
            uav_basic.step()
        


        #! Auto_uav step 
        self.auto_uav.step(acceleration, heading_correction) #! this will be created inside the reset method
        
        
        
        #! WE DO NOT NEED TO PERFORM HAS_LEFT_START_VERTIPORT and HAS_REACHED_END_VERTIPORT 
        #! because once the auto uav reaches its end_vertiport the training stops and we reset the environment
        
       
        obs = self._get_obs()
        reward = self.get_reward(obs)
        info = self._get_info()
        
        # Logic for termination and truncation
        auto_uav_current_position = self.auto_uav.current_position
        auto_uav_end_position = self.auto_uav.end_point

        distance_to_end_point = auto_uav_current_position.distance(auto_uav_end_position)
        
        if distance_to_end_point < self.auto_uav.landing_proximity: #!collect the clearing distance from atc or airspace or some other module - need to check 
            reached_end_vertiport = True
        else:
            reached_end_vertiport = False


        if reached_end_vertiport:
            terminated = True
        else:
            terminated = False

        #! check collision with static object 
        collision_static_obj = self.collision_with_static_obj()
        #! check collision with dynamic object 
        collision_dynamic_obj = self.collision_with_dynamic_obj()
        
        collision_detected = collision_static_obj or collision_dynamic_obj

        if collision_detected:
            truncated = True
        else:
            truncated = False
        
        

        return obs, reward, terminated, truncated, info


    def get_reward(self,obs) -> float:
        
        punishment_existing = 10000 # change to original punishment 

        if obs['intruder_detected'] == 0: 
            punishment_closeness:float = 0.0
        else:
            normed_nmac_distance = self.auto_uav.nmac_radius / self.auto_uav.detection_radius #what is this and why do i need it 
            punishment_closeness = -np.exp((normed_nmac_distance - obs['distance_to_intruder']) * 10)
        
        reward_to_destination = float(obs['agent_velocity']) * float(np.cos(obs['agent_deviation']))

        punishment_deviation = float(2 * (obs['agent_deviation'] / np.pi)**2) # change to original punishment 

        print(f'punishment existing: {punishment_existing}, punishment closeness: {punishment_closeness}, reward to destination: {reward_to_destination}, punishment deviation: {punishment_deviation}')
        reward_sum = punishment_existing + punishment_closeness + punishment_deviation + reward_to_destination
        print(f'reward sum before timestep: {reward_sum}')
        reward_sum *= self.current_time_step
        print(f'reward sum: {reward_sum}')
        
        return reward_sum




    

    def render_init(self,):
        fig, ax = plt.subplots()
        return fig, ax
    
    def render_static_assest(self, ax):
        self.airspace.location_utm_gdf.plot(ax=ax, color='gray', linewidth=0.6)
        self.airspace.location_utm_hospital_buffer.plot(ax=ax, color='green', alpha=0.3)
        self.airspace.location_utm_hospital.plot(ax=ax, color='black')
        #adding vertiports to static plot
        gpd.GeoSeries(self.sim_vertiports_point_array).plot(ax=ax, color='black')


    def render(self,fig, ax):
        plt.cla()
        self.render_static_assest(ax)
        
        # uav_basic PLOT LOGIC
        for uav_obj in self.uav_basic_list:
            uav_footprint_poly = uav_obj.uav_polygon_plot(uav_obj.uav_footprint)
            uav_footprint_poly.plot(ax=ax, color=uav_obj.uav_footprint_color, alpha=0.3)

            uav_nmac_poly = uav_obj.uav_polygon_plot(uav_obj.nmac_radius)
            uav_nmac_poly.plot(ax=ax, color=uav_obj.uav_nmac_radius_color, alpha=0.3)

            uav_detection_poly = uav_obj.uav_polygon_plot(uav_obj.detection_radius)
            uav_detection_poly.plot(ax=ax, color=uav_obj.uav_detection_radius_color,alpha=0.3)

        #TODO - render AUTO_UAV here
        auto_uav_footprint_poly = self.auto_uav.uav_polygon_plot(self.auto_uav.collision_radius)
        auto_uav_footprint_poly.plot(ax=ax, color = self.auto_uav.uav_footprint_color, alpha = 0.3)

        auto_uav_nmac_poly = self.auto_uav.uav_polygon_plot(self.auto_uav.nmac_radius)
        auto_uav_nmac_poly.plot(ax=ax, color=self.auto_uav.uav_nmac_radius_color, alpha=0.3)

        auto_uav_detection_poly = self.auto_uav.uav_polygon_plot(self.auto_uav.detection_radius)
        auto_uav_detection_poly.plot(ax=ax, color=self.auto_uav.uav_detection_radius_color,alpha=0.3)


        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(self.sleep_time)

    def _render_frame(self,):
        pass

    def close(self,):
        pass
    
    def get_start_vertiport_auto_uav(self,):
        for vertiport in self.atc.vertiports_in_airspace:
            if len(vertiport.uav_list) == 0:
                start_vertiport_auto_uav = vertiport
        return start_vertiport_auto_uav
    
    def get_end_vertiport_auto_uav(self,start_vertiport:Vertiport):
        some_vertiport = self.atc.provide_vertiport()
        while some_vertiport.location == start_vertiport.location:
            some_vertiport = self.atc.provide_vertiport()
        return some_vertiport
        
    def collision_with_static_obj(self,) -> bool:
        collision_with_static_obj, _ = self.auto_uav.get_state_static_obj(self.airspace.location_utm_hospital.geometry,'collision')
        return collision_with_static_obj
    
    def collision_with_dynamic_obj(self,) -> bool:
        collision = self.auto_uav.has_uav_collision(self.uav_basic_list)
        return collision
    
    def nmac_with_dynamic_obj(self,):
        nmac_info_dict = self.auto_uav.get_state_dynamic_obj(self.uav_basic_list,'nmac')
        return nmac_info_dict 



        #TODO - determine if one run of experiment will end when auto_uav reaches its first destination, or should we define a number of destinations or should it be a number of steps based completion 
        #! auto uav_basic will also need these two methods for moving to the next vertiport 
        #self.atc.has_left_start_vertiport(uav_basic) -> will need these two depending on how an experiment ends 
        #self.atc.has_reached_end_vertiport(uav_basic)