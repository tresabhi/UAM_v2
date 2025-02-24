import math
from dynamics_template import DynamicsTemplate
from shapely import Point
from uav_v2_template import UAV_v2_template



class ORCA_Dynamics(DynamicsTemplate):
    def __init__(self, dt=0.1, is_learning=False):
        super().__init__(dt, is_learning)


    def update(self, uav:UAV_v2_template, action):
        # action coming from orca
        # is a vector delta_vx, delta_vy
        
        new_v_x = action[0]
        new_v_y = action[1]
        new_speed = math.sqrt(new_v_x**2 + new_v_y**2)
        new_heading = math.atan2(new_v_y, new_v_x)

        _x = uav.current_position.x + (new_v_x * self.dt) #+ (0.5 * a_x * self.dt**2)
        _y = uav.current_position.y + (new_v_y * self.dt) #+ (0.5 * a_y * self.dt**2)

        uav.current_position = Point(_x, _y)
        uav.current_speed = new_speed
        uav.current_heading = new_heading

        
