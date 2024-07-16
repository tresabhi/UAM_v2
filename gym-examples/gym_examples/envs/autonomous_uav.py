#Smart uav, will accept action from RL algorithm 
from uav import UAV
from geopandas import GeoSeries
class Autonomous_UAV(UAV):
    '''Sub class of UAV, a smart UAV .
    It will accept acceleration and heading from RL algorithm'''

    def __init__(self,
                 start_vertiport,
                 end_vertiport,
                 landing_proximity = 50,
                 max_speed = 40):
        '''Representation of UAV in airspace. UAV motion represented in 2D plane. 
     Object is to move from start vertiport to end vertiport.
     A UAV instance requires a start and end vertiport.
     '''

        super().__init__(start_vertiport, end_vertiport, landing_proximity, max_speed)
        #! need to update the rendering properties, 
        #! since i have called super how do i pass/update the attributes
        #UAV rendering-representation properties 
        self.uav_footprint_color = 'black' # this color represents the UAV object 
        self.uav_nmac_radius_color = 'purple'
        self.uav_detection_radius_color = 'blue'
        self.uav_collision_controller = None

    

    def _update_speed(self, acceleration_from_controller, d_t=1):
        self.current_speed += acceleration_from_controller * d_t

    def step(self, acceleration, heading_correction):
        self._update_speed(acceleration,d_t=1)
        self._update_position(d_t=1)
        
        self._update_theta_d(heading_correction)
        self._update_ref_final_heading()
        pass
    
    



# from vertiport import Vertiport
# from shapely import Point
# import numpy as np 
# start = Vertiport(Point(0,0))
# end = Vertiport(Point(10,20))
# auto = Autonomous_UAV(start,end)

# print(np.array([auto.current_heading_deg - auto.current_ref_final_heading_deg]).shape)
