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
                 start_vertiport ,  
                 end_vertiport,
                 landing_proximity = 50., 
                 max_speed = 79,
                 ):
        
        #UAV builtin properties 
        self.heading_deg = np.random.randint(-178,178) + np.random.rand() # random heading between -180 and 180
        self.uav_footprint = 17 #H175 nose to tail length of 17m,
        self.nmac_radius = 150 #NMAC radius
        self.detection_radius = 550
        self.uav_footprint_color = 'blue'
        self.uav_nmac_radius_color = 'orange'
        self.uav_detection_radius_color = 'green'
        self.uav_collision_controller = None
        
        #UAV technical properties
        self.id = id(self)
        self.current_speed = 0
        self.max_speed:float = max_speed
        self.max_acceleration = 1 # m/s^2
        self.landing_proximity = landing_proximity 
        
        # UAV collision controller
        #TODO - the collision controller should be added here
        self.collision_controller = None

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
        self.current_heading_deg:float = self.heading_deg
        self.current_heading_radians = np.deg2rad(self.current_heading_deg)
        
        #Final heading calculation
        self.current_ref_final_heading_rad = np.arctan2(self.end_point.y - self.current_position.y, 
                                                                    self.end_point.x - self.current_position.x)
        self.current_ref_final_heading_deg = np.rad2deg(self.current_ref_final_heading_rad)


    def uav_polygon(self, dimension):
        return GeoSeries(self.current_position).buffer(dimension).iloc[0]
    
    def uav_polygon_plot(self, dimension):
        return GeoSeries(self.current_position).buffer(dimension)
    
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
                     

    def find_other_uav(self, uav_list):
        other_uav_list = []
        for uav in uav_list:
            if uav.id != self.id:
                other_uav_list.append(uav)
        return other_uav_list

    def calculate_intruder(self,uav_list):
        '''
        Here the self.intruder_uav_list is created everytime as an empty list, 
        So everystep this attribute is an empty list and its populated with uavs that are within detection radius.
        There is no return from this method, the data is stored in the attribute and should be accessed immediately after calling this method.
        Any subsequent routines can call the attribute and use the attribute for data processing 
        '''
        self.intruder_uav_list = []
        other_uav_list = self.find_other_uav(uav_list)

        for other_uav in other_uav_list:
            if self.uav_polygon(self.detection_radius).intersects(other_uav.uav_polygon(other_uav.detection_radius)):
                self.intruder_uav_list.append(other_uav)
        return len(self.intruder_uav_list)

    def calculate_intruder_distance(self, ):
        self.distance_to_intruder = []
        for other_uav in self.intruder_uav_list:
            self.distance_to_intruder.append(self.current_position.distance(other_uav.current_position))
        #TODO - should this list be organised based on distance 

    def calculate_intruder_speed(self,):
        self.intruder_speed_list = []
        for other_uav in self.intruder_uav_list:
            self.intruder_speed_list.append(abs(self.current_speed - other_uav.current_speed))
    
    def calculate_intruder_heading(self,):
        self.intruder_heading = []
        for other_uav in self.intruder_uav_list:
            self.intruder_heading.append(abs(self.current_heading_deg - other_uav.current_heading_deg))

    
    def get_state(self,uav_list ):
        deviation = self.current_heading_deg - self.current_ref_final_heading_deg
        speed = self.current_speed
        num_intruder = self.calculate_intruder(uav_list)
        intruder_distance = self.calculate_intruder_distance()
        intruder_speed = self.calculate_intruder_speed()
        intruder_heading = self.calculate_intruder_heading()

        self.state = {'deviation':deviation,
                      'speed':speed,
                      'num_intruder':num_intruder,
                      'intruder_distance':intruder_distance,
                      'intruder_speed':intruder_speed,
                      'intruder_heading':intruder_heading}
        
            


    def step(self, uav_list):
        '''Updates the position of the UAV.'''

        self._update_position(d_t=1, ) 
        self._update_speed(d_t=1)
        self._update_ref_final_heading()
        self._heading_correction()
        self.get_state(uav_list)
        


    




        
    
    





