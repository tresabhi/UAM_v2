from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import numpy as np 
from uav_v2_template import UAV_v2_template
class SensorTemplate(ABC):

    '''
    Collect other UAVs state information
    '''
    #FIX: 
    # set_data needs to be updated, currently we are only focused on dynamic obj, ie other_UAVs,
    # a sensor will sense any object, dynamic or static. 
    # proposed design - 
    # if intersection with object
    #   sensor will add label
    #       if sensor_label == UAV:
    #           add UAV to otherUAV_data_list
    #       else if sensor_label == restricted_airspace:
    #           add building to restricted_airspace_list
    #       else : 
    #           raise ERROR - this program at the moment should have these two objects only
    # 
    # 
    # The above structure is flexible for accepting different kinds of objects, using a label-string
    # and adding the object to a list for variable named label-string 



    @abstractmethod
    def __init__(self, space)->None:
        self.space = space
        self.uav_data = []
        self.restricted_airspace_data = []


    # UAV     
    #FIX: assign return signature
    def get_uav_detection():
        pass 
   
    @abstractmethod
    def get_nmac(self):
        '''Collect the time step and UAVs with who there was an NMAC'''
        pass
    
    @abstractmethod
    def get_uav_collision(self,):
        '''returns a bool if there is a collision along with UAV id'''
        pass


    # Restricted Airspace (ra)
    def set_ra_data(self,)->None:
        '''collect information about restricted airspace in space and save in data'''
        pass

    def get_ra_data()->List:
        '''return a list of restricted airspaces'''
        pass
   
    def get_ra_detection()->Tuple[bool, Dict]:
        pass

    def get_ra_collision()-> Tuple[bool, Dict]:
        pass


    


    # Deactivate sensor 
    def deactivate_nmac(self, uav:UAV_v2_template)->None:
        if uav.current_position.distance(uav.start) <= 100 or uav.current_position.distance(uav.end)<=100:
            return False, []

    def deactivate_detection()->None:
        pass

    def deactivate_collision(self, uav:UAV_v2_template)->None:
        if uav.current_position.distance(uav.start) <= 100 or uav.current_position.distance(uav.end)<=100:
            return False, [] 

