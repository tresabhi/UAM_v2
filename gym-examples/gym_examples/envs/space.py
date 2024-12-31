from typing import List
from uav_v2 import UAV_v2


class Space:
    def __init__(self):
        self.uav_list:List[UAV_v2] = []
        self.vertiport_list = []

    def set_vertiport(self,vertiport):
        self.vertiport_list.append(vertiport)
        return None
    
    def set_uav(self, uav):
        self.uav_list.append(uav)
        return None

    def get_vertiport_list(self):
        return self.vertiport_list
    
    def get_uav_list(self):
        return self.uav_list
    
