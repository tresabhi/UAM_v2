import math
from dynamics_template import DynamicsTemplate
from shapely import Point
from uav_v2_template import UAV_v2_template


class PointMassDynamics(DynamicsTemplate):

    def __init__(self, dt=0.1, is_learning=False):
        super().__init__(dt, is_learning)


    def update(self,uav:UAV_v2_template, action):
        # action[0] -> acceleration
        # action[1] -> heading_change 

        if self.is_learning:
            action[0] = 1 * action[0] # some_re_normalization
            action[1] = math.pi * action[1] # some_re_normalization

        if uav.mission_complete_status:

            #UAV has reached its destination
            uav.current_position = uav.end
        
        
        else:
            
            new_speed = uav.current_speed + action[0] * self.dt
            new_heading = uav.current_heading + action[1]

            v_x = new_speed * math.cos(new_heading)
            a_x = action[0] * math.cos(new_heading)
            
            v_y = new_speed * math.sin(new_heading)
            a_y = action[0] * math.sin(new_heading)

            _x = uav.current_position.x + (v_x * self.dt) + (0.5 * a_x * self.dt**2)
            _y = uav.current_position.y + (v_y * self.dt) + (0.5 * a_y * self.dt**2)

            uav.current_position = Point(_x, _y)
            uav.current_speed = new_speed
            uav.current_heading = new_heading




        
        
