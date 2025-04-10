from sensor_template import SensorTemplate
from typing import List, Dict, Tuple, Any
from pandas import DataFrame
import numpy as np
import shapely
from uav_v2_template import UAV_v2_template
# from utils import compute_time_to_impact

class UniversalSensor(SensorTemplate):

    def __init__(self, space):
        super().__init__(space)


    def set_data(self, self_uav:UAV_v2_template) -> None:
        '''Collect data of other UAVs in space'''
        # get self_uav detection radius 
        self.detection_radius = self_uav.detection_radius
        # get UAV list from space
        uav_list:List[UAV_v2_template] = self.space.get_uav_list()
        

        for uav in uav_list:
            #! when there is no agent within the detection radius, other_uav_states are NONE
            if uav.id != self_uav.id:
                if self_uav.current_position.distance(uav.current_position) <= self.detection_radius:
                    other_uav_data = {'other_uav id': uav.id,
                            'other_uav_current_position':uav.current_position,
                            'other_uav_current_speed':uav.current_speed,
                            'other_uav_current_heading':uav.current_heading,
                            'other_uav_radius': uav.radius}
                    self.data.append(other_uav_data)
        return None
    

    
    def get_data(self) -> List[Dict]:
        return self.data
       
    def get_nmac(self, self_uav:UAV_v2_template) -> Tuple[bool, List]:
        nmac_list = []
        uav_list = self.space.get_uav_list()

        for uav in uav_list:
            if uav.id == self_uav.id:
                continue
            else:
                if self_uav.current_position.buffer(self_uav.nmac_radius).intersects(uav.current_position.buffer(uav.nmac_radius)):
                    nmac_list.append(uav)
        if len(nmac_list) > 0:
            return True, nmac_list
        return False, nmac_list

    def get_collision(self, self_uav:UAV_v2_template) -> Tuple[bool, Tuple[Any, Any]]:
        uav_list = self.space.get_uav_list()
        
        for uav in uav_list:
            if uav.id == self_uav.id:
                continue
            else:
                if self_uav.current_position.buffer(self_uav.radius).intersects(uav.current_position.buffer(uav.radius)):
                    return True, (self_uav.id, uav.id)
        return False, None

            