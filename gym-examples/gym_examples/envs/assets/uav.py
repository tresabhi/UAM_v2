#Deterministic uavs

import numpy as np
from shapely.geometry import Point
from geopandas import GeoSeries
from vertiport import Vertiport
from collision_controller import Collision_controller
#TODO - abstract controller, basic collision controller 
#from collision_avoidance_controller_basic import uav_collision_detection, uav_nmac_detection, static_collision_detection, static_nmac_detection


class UAV:
    '''Representation of UAV in airspace. UAV translates in 2D plane, 
     object is to move from start vertiport to end vertiport.
     This object has builtin collision avoidance mechanism.'''
    def __init__(self,
                 start_vertiport , #TODO - remove start and end vertiport, assign them later 
                 end_vertiport,
                 landing_proximity = 50., 
                 max_speed = 79., # Airbus H175 - max curise speed 287 kmph - 79 meters per second
                 heading_deg = np.random.randint(-178,178)+np.random.rand(), # random heading between -180 and 180 #TODO - this can be assigned later 
                 uav_footprint = 17, #H175 nose to tail length of 17m,
                 nmac_radius = 150, #NMAC radius
                 detection_radius = 250,
                 uav_footprint_color = 'blue',
                 uav_nmac_radius_color = 'orange',
                 uav_detection_radius_color = 'green',
                 uav_collision_controller = None
                 ):
        #TODO - add buffer area attribute for UAV(object collision), and NMAC
        #UAV technical properties
        self.id = id(self)
        self.current_speed = 0
        self.max_speed:float = max_speed
        self.max_acceleration = 1 # m/s^2
        self.landing_proximity = landing_proximity 
        self.collision_controller = uav_collision_controller

        #UAV footprint and NMAC
        self.uav_footprint = uav_footprint # this value is used to draw a circle around uavs current position
        self.nmac_radius =  nmac_radius # same
        self.detection_radius = detection_radius #TODO - may1, 1153, start from here, next - develop the detection system for uav and create states
        self.uav_footprint_color = uav_footprint_color
        self.uav_nmac_radius_color = uav_nmac_radius_color
        self.uav_detection_radius_color = uav_detection_radius_color

        #UAV soft properties
        self.leaving_start_vertiport = False
        self.reaching_end_vertiport = False

        #Vertiport assignement
        self.start_vertiport = start_vertiport
        self.end_vertiport = end_vertiport
        
        #UAV position properties
        self.start_point:Point = self.start_vertiport.location
        self.end_point:Point = self.end_vertiport.location
        self.current_position:Point = self.start_point
        
        #UAV heading properties
        self.current_heading_deg:float = heading_deg
        self.current_heading_radians = np.deg2rad(self.current_heading_deg)
        
        #Final heading calculation
        self.current_ref_final_heading_rad = np.arctan2(self.end_point.y - self.current_position.y, 
                                                                    self.end_point.x - self.current_position.x)
        self.current_ref_final_heading_deg = np.rad2deg(self.current_ref_final_heading_rad)

        #UAV detection properties
        self.detected_uav_list = []
        self.detected_airspace_list = []


        #UAV polygon object 
        #TODO - delete if not needed
        # self.uav_object = GeoSeries(self.current_position)
        # self.uav_self = self.uav_object.buffer(self.uav_footprint)
        # self.uav_self = self.uav_object.buffer(self.nmac_radius)



    def reset_uav(self, ):
        '''Update the start vertiport of a UAV 
        to a new start vertiport, 
        argument of this method '''
        self.leaving_start_vertiport = False
        self.reaching_end_vertiport = False 


    def update_end_point(self,):
        '''Updates the UAV end point using its own end_vertiport location'''
        self.end_point = self.end_vertiport.location

    def update_start_point(self,):
        self.start_point = self.start_vertiport.location

    
    def _update_position(self,d_t:float,):
        '''Internal method. Updates current_position of the UAV after d_t seconds.
           This uses a first order Euler's method to update the position.
           '''
        self.acceleration_controller()
        update_x = self.current_position.x + self.current_speed * np.cos(self.current_heading_radians) * d_t 
        update_y = self.current_position.y + self.current_speed * np.sin(self.current_heading_radians) * d_t 
        self.current_position = Point(update_x,update_y)
        
    
    #TODO - need to see what this polygon looks like  
    def uav_footprint_polygon(self, ):
        return GeoSeries(self.current_position).buffer(self.uav_footprint)
    
    def uav_nmac_polygon(self,):
        return GeoSeries(self.current_position).buffer(self.nmac_radius)
    
    
    def _update_speed(self,d_t, ):
        self.acceleration_controller()
        if self.current_position.distance(self.end_point) <= 700:
            self.current_speed = self.current_speed + self.current_acceleration
        else:
            if self.current_speed < self.max_speed:
                self.current_speed = self.current_speed + (0.5)*self.current_acceleration*d_t
            else:
                self.current_speed = self.max_speed
        

    def acceleration_controller(self,):
        if self.current_position.distance(self.end_point) < 1500:
            self.current_acceleration = -2*self.max_acceleration
        else:
            self.current_acceleration = self.max_acceleration


    def _update_ref_final_heading(self, ): 
        '''Internal method. Updates the heading of the aircraft, pointed towards end_point'''
        self.current_ref_final_heading_rad = np.arctan2(self.end_point.y - self.current_position.y, 
                                                        self.end_point.x - self.current_position.x)
        self.current_ref_final_heading_deg = np.rad2deg(self.current_ref_final_heading_rad)
        
        

    def _heading_correction(self, ): 
        '''Internal method. Updates heading of the aircraft, pointed towards ref_final_heading_deg''' 
        
        avg_rate_of_turn = 20 #degree, collected from google - https://skybrary.aero/articles/rate-turn#:~:text=Description,%C2%B0%20turn%20in%20two%20minutes.

        #! need to find how to dynamically slow down the turn rate as we get close to the ref_final_heading
        if np.abs(self.current_ref_final_heading_deg - self.current_heading_deg) < avg_rate_of_turn:
            avg_rate_of_turn = 1 #degree Airbus H175

        if np.abs(self.current_ref_final_heading_deg - self.current_heading_deg) < 0.5:
            avg_rate_of_turn = 0.
        
        #* logic for heading update 
        if (np.sign(self.current_ref_final_heading_deg)==np.sign(self.current_heading_deg)==1):
            # and (ref_final_heading > current_heading_deg)) or ((np.sign(ref_final_heading)==np.sign(current_heading_deg)== -1) and (np.abs(ref_final_heading)<(np.abs(current_heading_deg)))):
            if self.current_ref_final_heading_deg > self.current_heading_deg:
                self.current_heading_deg += avg_rate_of_turn #counter clockwise turn
                self.current_heading_radians = np.deg2rad(self.current_heading_deg) 
            elif self.current_ref_final_heading_deg < self.current_heading_deg:
                self.current_heading_deg -= avg_rate_of_turn #clockwise turn
                self.current_heading_radians = np.deg2rad(self.current_heading_deg)
            else:
                pass  
        
        elif np.sign(self.current_ref_final_heading_deg) == np.sign(self.current_heading_deg) == -1:
            if np.abs(self.current_ref_final_heading_deg) < np.abs(self.current_heading_deg):
                self.current_heading_deg += avg_rate_of_turn #counter clockwise turn
                self.current_heading_radians = np.deg2rad(self.current_heading_deg)
            elif np.abs(self.current_ref_final_heading_deg) > np.abs(self.current_heading_deg):
                self.current_heading_deg -= avg_rate_of_turn #clockwise turn
                self.current_heading_radians = np.deg2rad(self.current_heading_deg)
            else:
                pass
                
        elif np.sign(self.current_ref_final_heading_deg) == 1 and np.sign(self.current_heading_deg) == -1:
            self.current_heading_deg += avg_rate_of_turn #counter clockwise turn
            self.current_heading_radians = np.deg2rad(self.current_heading_deg)

        elif np.sign(self.current_ref_final_heading_deg) == -1 and np.sign(self.current_heading_deg) == 1:
            self.current_heading_deg -= avg_rate_of_turn #clockwise turn
            self.current_heading_radians = np.deg2rad(self.current_heading_deg)

        else:
            raise Exception('Error in heading correction')
        

    


    # def uav_collision_detection(self, uav_list): #TODO - uav_list argument includes self, thats why I am observing 'Collision Detected' continuously
    #     '''
    #     This procedure has to be performed first 
    #     if distance of UAV from vertiport less than equal to some value
    #         DO NOT PERFORM collision avoidance 
    #     THEN - i can perform what I have below
    #     '''
    #     modified_uav_list = []
    #     for uav in uav_list:
    #         if self.id != uav.id:
    #             modified_uav_list.append(uav)
        
    #     uav_footprint_polygon = GeoSeries(self.current_position).buffer(self.uav_footprint).iloc[0]
    #     for uav_other in modified_uav_list:
    #         if uav_footprint_polygon.intersects(GeoSeries(uav_other.current_position).buffer(uav_other.uav_footprint).iloc[0]):
    #             print('UAV Collision Detected')


    # def uav_nmac_detection(self, uav_list): #TODO - uav_list argument includes self, thats why I am observing 'Collision Detected' continuously
    #     '''
    #     This procedure has to be performed first 
    #     if distance of UAV from vertiport less than equal to some value
    #         DO NOT PERFORM collision avoidance 
    #     THEN - i can perform what I have below
    #     '''
    #     modified_uav_list = []
    #     for uav in uav_list:
    #         if self.id != uav.id:
    #             modified_uav_list.append(uav)
        
    #     uav_self = GeoSeries(self.current_position).buffer(self.nmac_radius).iloc[0]
    #     for uav_other in modified_uav_list:
    #         if uav_self.intersects(GeoSeries(uav_other.current_position).buffer(uav_other.nmac_radius).iloc[0]):
    #             #TODO - this is not a good implementation - needs a good fix 
    #             self.current_heading_deg = np.random.randint(low=-45, high=46)
    #             self.current_heading_radians = np.deg2rad(self.current_heading_deg)
        

    # def static_collision_detection(self, static_object_df:GeoSeries):
    #     # check intersection with uav list - here return is true or false, true meaning intersection 
    #     # 
    #     # check intersection with raz_list
    #     uav_footprint_polygon = GeoSeries(self.current_position).buffer(self.uav_footprint).iloc[0]

    #     for i in range(len(static_object_df)):
    #         if uav_footprint_polygon.intersects(static_object_df.geometry.iloc[i]):
    #             print('Static object collision')

    # def static_nmac_detection(self, static_object_df) : #static_object_df -> dataframe  # return contact_uav_id 
    #     # check intersection with uav list -  return is geoseries with true or false, true meaning intersection with contact_uav 
    #     # collect contact_uav id for true in geoseries
    #     # use the contact_uav id to collect information of the uav - 
    #     # required info 
    #     #                contact_uav - heading, distance from contact_uav(can be calculated using position), velocity
    #     #                ownship_uav     - deviation, velocity, has_intruder
    #     #                relative bearing - calculate as -> ownship_heading - absolute_angle_between_contact_and_ownship
        
    #     # check intersection with static_object_df ??  

    #     uav_self = GeoSeries(self.current_position).buffer(self.nmac_radius).iloc[0]
    #     # print('type: ', type(uav_self.iloc[0]))

    #     for i in range(len(static_object_df)):
    #         if uav_self.intersects(static_object_df.iloc[i]):
    #             # 90 degree clockwise rotation 
    #             #TODO - need to update this NOW !!!
    #             self.current_heading_deg = self.current_ref_final_heading_deg - 45
    #             self.current_heading_radians = np.deg2rad(self.current_heading_deg)

    
    #TODO - test this method, check output of this method and ensure the self.detected_uav_list is getting updated everystep 
    def uav_detection(self,uav_list):
        '''
        this method will be called over and over again with a changing stepindex value 
        an array that will hold uavs that are detected.
        this array will update each step.
        if there is no uav detected at any given step then the array should be empty.
        '''
        detected_uav_list = []
        modified_uav_list = []
        for uav in uav_list:
            if self.id != uav.id:
                modified_uav_list.append(uav)
    
        uav_self = GeoSeries(self.current_position).buffer(self.detection_radius).iloc[0]
        
        for uav_other in modified_uav_list:
            if uav_self.intersects(GeoSeries(uav_other.current_position).buffer(uav_other.detection_radius).iloc[0]):
                detected_uav_list.append(uav_other)
        
        self.detected_uav_list = detected_uav_list








        
            
        
        
    #TODO - test this method   
    def contact_uav_information(self,):
        # using contact_uav_id collect the following 
        #   contact_uav - heading, position, velocity, 
        
        uav_info_dict = dict()
        if len(self.detected_uav_list) == 0:
            return None
        else:
            for other_uav in self.detected_uav_list:
                uav_info_dict[other_uav.id] = (other_uav.current_position,
                                               other_uav.current_heading_deg,
                                               other_uav.current_ref_final_heading_deg)

            

    def state_observation(self, ):
        # Here return a list with the following states
        # state -> [deviation = (self.current_heading - self.ref_final_headin)
        #           speed, 
        #           current heading, 
        #           has_contact,      NOT SURE WHY THIS IS IMPORTANT, AND HOW THIS IS USED IN THE RL FRAMEWORK
        #           contact heading,           
        #           distance from contact np.abs(self.current_position - contact_uav.current_position)
        #           contact speed
        #           relative bearing 
        pass    


    def step(self, location,location_buffer, uav_list):
        '''Updates the position of the UAV.'''


        if self.collision_controller == None:
            self._update_position(d_t=1, ) 
            self._update_speed(d_t=1)
            self._update_ref_final_heading()
            self._heading_correction()
        elif isinstance(self.collision_controller, Collision_controller):
            self._update_position(d_t=1, ) 
            self._update_speed(d_t=1)
            self._update_ref_final_heading()
            self._heading_correction()
            self.collision_controller.static_nmac_detection(self,location_buffer)
            self.collision_controller.uav_nmac_detection(self, uav_list)
            self.collision_controller.uav_collision_detection(self, uav_list)
            self.collision_controller.static_collision_detection(self, location)
            

            #print('uav: ', self.id, 'current position: ', self.current_position)
    




        
    
    





