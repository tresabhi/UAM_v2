from typing import List
from uav_v2_template import UAV_v2_template


class Space:
    def __init__(self):
        self.uav_list:List = []
        self.vertiport_list:List = []

    def set_vertiport(self,vertiport):
        self.vertiport_list.append(vertiport)
        return None
    
    def set_uav(self, uav:UAV_v2_template):
        self.uav_list.append(uav)
        return None

    def get_vertiport_list(self):
        return self.vertiport_list
    
    def get_uav_list(self) -> List[UAV_v2_template]:
        return self.uav_list
    
