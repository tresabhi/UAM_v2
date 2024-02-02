import numpy as np
from shapely.geometry import Point

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
        
        

    def __repr__(self):
        return ('UAV({id}), start_point: {start}, end_point:{end}, current_point: {current}, current_heading: {current_heading}, final_heading: {final_heading}'
                .format(id = self.id, start = self.start_point, end =self.end_point ,current = self.current_position,current_heading = self.current_heading_deg, final_heading = self.current_ref_final_heading_deg))
    
    
    #! might need to turn this into an internal function
    #! this method has side effects - no returns
    def update_position(self,d_t:float, ):
        '''Updates current_position of the UAV after d_t seconds.'''

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
        

    def step(self,):
        '''Updates the position of the UAV.'''
        
        
        self.update_position(d_t=1) #seconds
        self.update_ref_final_heading()
        self.heading_correction()
        
        #! need to add a check for reaching the end point
        
    
        




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

        
    
    





