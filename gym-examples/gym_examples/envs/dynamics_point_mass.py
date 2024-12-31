from dynamics_template import DynamicsTemplate
from shapely import Point

class PointMassDynamics(DynamicsTemplate):

    def __init__(self, dt=0.1):
        super().__init__(dt)


    def update(self,uav, action):
        
        new_speed = uav.current_speed + action[0] * self.dt
        new_heading = uav.current_heading + action[1]

        _x = uav.current_position.x + (uav.current_speed * self.dt) + (0.5 * action[0] * self.dt**2)
        _y = uav.current_position.y + (uav.current_speed * self.dt) + (0.5 * action[0] * self.dt**2)

        uav.current_position = Point(_x, _y)
        uav.current_speed = new_speed
        uav.current_heading = new_heading

        
        
