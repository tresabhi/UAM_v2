from abc import ABC, abstractmethod
from typing import Tuple

class ControllerTemplate(ABC):

    @abstractmethod
    def __call__(self, observation) -> Tuple:
        '''Returns action based on observation.
        
            Args: observation array

            Returns: action tuple
        
        '''

        pass

