import math
from dynamics_template import DynamicsTemplate
from shapely import Point
from uav_v2_template import UAV_v2_template


class PointMassDynamics(DynamicsTemplate):

    def __init__(self, dt=0.1):
        super().__init__(dt)


    def update(self,uav:UAV_v2_template, action):

        if uav.mission_complete_status:

            #UAV has reached its destination
            uav.current_position = uav.end
        
        
        else:
            
            new_speed = uav.current_speed + action[0] * self.dt
            new_heading = uav.current_heading + action[1]

            v_x = uav.current_speed * math.cos(new_heading)
            a_x = action[0] * math.cos(new_heading)
            
            v_y = uav.current_speed * math.sin(new_heading)
            a_y = action[0] * math.sin(new_heading)

            _x = uav.current_position.x + (v_x * self.dt) + (0.5 * a_x * self.dt**2)
            _y = uav.current_position.y + (v_y * self.dt) + (0.5 * a_y * self.dt**2)

            uav.current_position = Point(_x, _y)
            uav.current_speed = new_speed
            uav.current_heading = new_heading




        
        
