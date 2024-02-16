#Deterministic uavs

import numpy as np
from shapely.geometry import Point
from geopandas import GeoSeries

class UAV:
    def __init__(self, 
                 start_point:Point, #need to be provided by the airspace for valid points
                 end_point:Point,  
                 speed = 79.0, # Airbus H175 - max curise speed 287 kmph - 79 meters per second
                 heading_deg = np.random.randint(-178,178)+np.random.rand(), # random heading between -180 and 180
                 ):
        #UAV technical properties
        self.id = id(self)
        self.speed:float = speed
        
        #UAV position properties
        self.start_point:Point = start_point
        self.end_point:Point = end_point
        self.current_position:Point = start_point
        
        #UAV heading properties
        self.current_heading_deg:float = heading_deg
        self.current_heading_radians = np.deg2rad(self.current_heading_deg)
        
        #Final heading calculation
        self.current_ref_final_heading_rad = np.arctan2(self.end_point.y - self.current_position.y, 
                                                                    self.end_point.x - self.current_position.x)
        self.current_ref_final_heading_deg = np.rad2deg(self.current_ref_final_heading_rad)

        #path trace of uav
        self.path_trace = []
        
        

    def __repr__(self):
        return f"UAV({self.start_point}, {self.end_point})"
    
    
    #! might need to turn this into an internal function
    #! this method has side effects - no returns
    def update_position(self,d_t:float, ):
        '''Updates current_position of the UAV after d_t seconds.
           This uses a first order Euler's method to update the position.
           '''

        update_x = self.current_position.x + d_t * self.speed * np.cos(self.current_heading_radians)
        update_y = self.current_position.y + d_t * self.speed * np.sin(self.current_heading_radians)
        
        self.current_position = Point(update_x,update_y)
        
    
    #! might need to turn this into an internal function
    #! this method has side effects - no returns
    def update_ref_final_heading(self, ): 
        '''Updates the heading of the aircraft, pointed towards end_point'''
        self.current_ref_final_heading_rad = np.arctan2(self.end_point.y - self.current_position.y, 
                                                        self.end_point.x - self.current_position.x)
        self.current_ref_final_heading_deg = np.rad2deg(self.current_ref_final_heading_rad)
        
        

    #! might need to turn this into an internal function
    #! this method has side effects - no returns
    def heading_correction(self, ): 
        '''Updates heading of the aircraft, pointed towards ref_final_heading_deg''' 
        
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
            raise Exception
        

    
    def collision_detection(self,uav_list:GeoSeries, raz_list:GeoSeries):
        # check intersection with uav list - here return is true or false, true meaning intersection 
        # 
        # check intersection with raz_list
        pass

    def nmac_detection(self, uav_list:GeoSeries, raz_list:GeoSeries) : # return contact_uav_id 
        # check intersection with uav list -  return is geoseries with true or false, true meaning intersection with contact_uav 
        # collect contact_uav id for true in geoseries
        # use the contact_uav id to collect information of the uav - 
        # required info 
        #                contact_uav - heading, distance from contactuav(can be calculated using position), velocity
        #                ownship_uav     - deviation, velocity, has_intruder
        #                relative bearing - calculate as -> ownship_heading - absolute_angle_between_contact_and_ownship
        
        # check intersection with raz_list
        pass
        
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
        
        
        self.update_position(d_t=1) #seconds
        self.update_ref_final_heading()
        self.heading_correction()
        

    def navigate_to_proximity(self,vertiport_proximity=20):
        '''This method navigates the uav to its end point, 
            the uav motion is terminated 
            vertiport_proximity distance away 
            from end point
        '''
        while ((np.abs(self.current_position.x - self.end_point.x)>vertiport_proximity) and (np.abs(self.current_position.y - self.end_point.y)>vertiport_proximity)):
            self.step() #TODO in the visual we see the uav move, actual NMAC and collision needs to be determined in step
            self.path_trace.append(self.current_position)
        
    
        




#* Test
        
# start_point = Point(0,0)
# end_point = Point(-1200,-1000)
# uav = UAV(start_point=start_point, end_point=end_point, speed=10.0)
# print(uav)

# while ((np.abs(uav.current_position.x - end_point.x)>5) and (np.abs(uav.current_position.y - end_point.y)>5)):
#     uav.step()
#     print(uav)

# # for i in range(100):
# #     uav.step()
# #     print(uav)
    

# print('Reached destination:', uav.current_position)

        
    
    





