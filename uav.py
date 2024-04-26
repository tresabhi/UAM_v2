#Deterministic uavs

import numpy as np
from shapely.geometry import Point
from geopandas import GeoSeries
from vertiport import Vertiport



class UAV:
    '''Representation of UAV in airspace. UAV translates in 2D plane, 
     object is to move from start vertiport to end vertiport.
     This object has builtin collision avoidance mechanism.'''
    def __init__(self,
                 start_vertiport ,
                 end_vertiport,
                 landing_proximity = 50, 
                 max_speed = 79, # Airbus H175 - max curise speed 287 kmph - 79 meters per second
                 heading_deg = np.random.randint(-178,178)+np.random.rand(), # random heading between -180 and 180
                 uav_footprint = 17, #H175 nose to tail length of 17m,
                 nmac_radius = 150, #NMAC radius
                 uav_footprint_color = 'blue',
                 uav_nmac_radius_color = 'orange'
                 ):
        #TODO - add buffer area attribute for UAV(object collision), and NMAC
        #UAV technical properties
        self.id = id(self)
        self.current_speed = 0
        self.max_speed:float = max_speed
        self.max_acceleration = 1 # m/s^2
        self.landing_proximity = landing_proximity

        #UAV footprint and NMAC
        self.uav_footprint = uav_footprint 
        self.nmac_radius =  nmac_radius
        self.uav_footprint_color = uav_footprint_color
        self.uav_nmac_radius_color = uav_nmac_radius_color

        #UAV soft properties
        self.left_start_vertiport = False
        self.reached_end_vertiport = False
        self.first_flight_of_day = True

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


        #UAV polygon object 
        #TODO - delete if not needed
        # self.uav_object = GeoSeries(self.current_position)
        # self.uav_nmac_polygon = self.uav_object.buffer(self.uav_footprint)
        # self.uav_nmac_polygon = self.uav_object.buffer(self.nmac_radius)



    def reset_uav(self, ):
        '''Update the start vertiport of a UAV 
        to a new start vertiport, 
        argument of this method '''
        #TODO - Reset soft properties back to original properties
        self.left_start_vertiport = False
        self.reached_end_vertiport = False 


    
    def update_end_point(self,):
        '''Updates the UAV end point using its own end_vertiport location'''
        self.end_point = self.end_vertiport.location

    #! Might not need this - check usage if any
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
        
    
    #TODO - these two methods are called over and over again, rethink logic 
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
        

    


    def uav_collision_detection(self, uav_list): #TODO - uav_list argument includes self, thats why I am observing 'Collision Detected' continuously
        '''
        This procedure has to be performed first 
        if distance of UAV from vertiport less than equal to some value
            DO NOT PERFORM collision avoidance 
        THEN - i can perform what I have below
        '''
        modified_uav_list = []
        for uav in uav_list:
            if self.id != uav.id:
                modified_uav_list.append(uav)
        
        uav_footprint_polygon = GeoSeries(self.current_position).buffer(self.uav_footprint).iloc[0]
        for uav_other in modified_uav_list:
            if uav_footprint_polygon.intersects(GeoSeries(uav_other.current_position).buffer(uav_other.uav_footprint).iloc[0]):
                print('UAV Collision Detected')


    def uav_nmac_detection(self, uav_list): #TODO - uav_list argument includes self, thats why I am observing 'Collision Detected' continuously
        '''
        This procedure has to be performed first 
        if distance of UAV from vertiport less than equal to some value
            DO NOT PERFORM collision avoidance 
        THEN - i can perform what I have below
        '''
        modified_uav_list = []
        for uav in uav_list:
            if self.id != uav.id:
                modified_uav_list.append(uav)
        
        uav_nmac_polygon = GeoSeries(self.current_position).buffer(self.nmac_radius).iloc[0]
        for uav_other in modified_uav_list:
            if uav_nmac_polygon.intersects(GeoSeries(uav_other.current_position).buffer(uav_other.nmac_radius).iloc[0]):
                self.current_heading_deg = 45
                self.current_heading_radians = np.deg2rad(self.current_heading_deg)
        

    def static_collision_detection(self, static_object_df:GeoSeries):
        # check intersection with uav list - here return is true or false, true meaning intersection 
        # 
        # check intersection with raz_list
        uav_footprint_polygon = GeoSeries(self.current_position).buffer(self.uav_footprint).iloc[0]

        for i in range(len(static_object_df)):
            if uav_footprint_polygon.intersects(static_object_df.geometry.iloc[i]):
                print('Static object collision')

    def static_nmac_detection(self, static_object_df) : #static_object_df -> dataframe  # return contact_uav_id 
        # check intersection with uav list -  return is geoseries with true or false, true meaning intersection with contact_uav 
        # collect contact_uav id for true in geoseries
        # use the contact_uav id to collect information of the uav - 
        # required info 
        #                contact_uav - heading, distance from contact_uav(can be calculated using position), velocity
        #                ownship_uav     - deviation, velocity, has_intruder
        #                relative bearing - calculate as -> ownship_heading - absolute_angle_between_contact_and_ownship
        
        # check intersection with static_object_df ??  

        uav_nmac_polygon = GeoSeries(self.current_position).buffer(self.nmac_radius).iloc[0]
        # print('type: ', type(uav_nmac_polygon.iloc[0]))

        for i in range(len(static_object_df)):
            if uav_nmac_polygon.intersects(static_object_df.iloc[i]):
                # 90 degree clockwise rotation 
                self.current_heading_deg = 45
                self.current_heading_radians = np.deg2rad(self.current_heading_deg)


        
        
    def contact_uav_information(self, contact_uav_id, uav_db):
        # using contact_uav_id collect the following 
        #   contact_uav - heading, position, velocity, 
        pass

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


    def step(self,):
        '''Updates the position of the UAV.'''
        self._update_position(d_t=1, ) 
        self._update_speed(d_t=1)
        self._update_ref_final_heading()
        self._heading_correction()
        

        #print('uav: ', self.id, 'current position: ', self.current_position)
    




        
    
    





