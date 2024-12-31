from abc import ABC, abstractmethod
import math
from shapely import Point

from sensor_template import SensorTemplate
from controller_template import ControllerTemplate
from dynamics_template import DynamicsTemplate

class UAV_v2_template(ABC):
    

    @abstractmethod
    def __init__(self, controller, dynamics, sensor, radius):
        # controller and dynamics need to be a valid pair
        self.id = id(self) 
        self.radius = radius
        
        self.controller:ControllerTemplate = controller
        self.dynamics:DynamicsTemplate = dynamics
        self.sensor:SensorTemplate = sensor


        self.current_speed = 0
        

    @abstractmethod
    def assign_start_end(self, start:Point, end:Point):
        self.start = start
        self.end = end
        
        self.current_position = start
        self.current_heading = math.atan2((start.y - end.y), (start.x - end.x))
        
        self.body = self.current_position.buffer(self.radius)

    def get_state(self,):
        return {'id':self.id, 'current_position':self.current_position, 'current_heading':self.current_heading}
    
    # Since dynamics, controller, and sensor are all individual components,
    # they should be used/called by the UAV inside its implementation

    
    @abstractmethod
    def get_sensor_data(self,):
        '''
        Collect sensor data
        '''

        self.sensor_data = self.sensor.get_data()
        
        return self.sensor_data
    
    
    @abstractmethod
    def get_action(self, observation):
        self.action = self.controller(observation=observation)
        return self.action

    @abstractmethod
    def step(self, action):
        # depending on controller the action will have different input arguments
        # depending on output of controller dynamics will update current position and current heading
        self.dynamics.update(self, action)
        return None 

    
    
    

    
