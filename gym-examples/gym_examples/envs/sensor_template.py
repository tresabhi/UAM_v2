from abc import ABC, abstractmethod
from typing import Dict, List

class SensorTemplate(ABC):

    '''
    Collect self and other UAVs state information
    '''
    @abstractmethod
    def __init__(self)->None:
        self.data = []



    @abstractmethod
    def set_data(self)->None:
        '''Collect information about other UAVs and save in data'''
        pass
    
    @abstractmethod
    def get_data(self)->List:
        return self.data
    
    



