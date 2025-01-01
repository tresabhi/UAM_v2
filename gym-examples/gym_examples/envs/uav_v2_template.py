from abc import ABC, abstractmethod
from typing import List, Dict
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
        self.sensor:SensorTemplate = sensor
        self.dynamics:DynamicsTemplate = dynamics
        self.controller:ControllerTemplate = controller
        self.mission_complete_distance = 50
        self.current_speed = 0
        
        

    @abstractmethod
    def assign_start_end(self, start:Point, end:Point):
        self.mission_complete_status = False
        self.start = start
        self.end = end
        self.current_position = start
        self.current_heading = math.atan2((start.y - end.y), (start.x - end.x))
        self.body = self.current_position.buffer(self.radius)
        return None
    
    @abstractmethod
    def get_mission_status(self,) -> bool:
        if self.current_position.distance(self.end) <= self.mission_complete_distance:
            misison_complete_status = True
        else:
            misison_complete_status = False
        return misison_complete_status
    
    @abstractmethod
    def set_mission_complete_status(self,mission_complete_status) -> None:
        self.mission_complete_status = mission_complete_status
        return None



    
    @abstractmethod
    def get_state(self,):
        return {'id':self.id, 'start':self.start, 'end':self.end, 'current_position':self.current_position, 'current_heading':self.current_heading}
    
    @abstractmethod
    def get_sensor_data(self,):
        '''
        Collect sensor data
        '''
        self.sensor_data = self.sensor.get_data()
        return self.sensor_data
    
    @abstractmethod
    def get_obs(self) -> List[Dict]:
        obs = []
        obs.append(self.get_state())
        obs = obs + self.get_sensor_data()
        return obs
    
    @abstractmethod
    def get_action(self, observation):
        self.action = self.controller(observation=observation)
        return self.action
    
