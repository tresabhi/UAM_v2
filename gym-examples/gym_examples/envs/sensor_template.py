from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np 
class SensorTemplate(ABC):

    '''
    Collect other UAVs state information
    '''
    @abstractmethod
    def __init__(self, space)->None:
        self.space = space
        self.data = []



    @abstractmethod
    def set_data(self)->None:
        '''Collect information about other UAVs in space and save in data'''
        return None
    
    @abstractmethod
    def get_data(self)->List[Dict]:
        '''Return observation data about other UAVs in space based on sorting criteria'''
        return self.data
    
    @abstractmethod
    def get_nmac(self):
        '''Collect the time step and UAVs with who there was an NMAC'''
        pass
    
    @abstractmethod
    def get_collision(self,):
        '''returns a bool if there is a collision along with UAV id'''
        pass