from abc import ABC, abstractmethod
from typing import Dict, List

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
        pass
    
    @abstractmethod
    def get_data(self)->List[Dict]:
        '''Return observation data about other UAVs in space'''
        return self.data
    
    



