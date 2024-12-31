from sensor_template import SensorTemplate
from uav_v2 import UAV_v2
from space import Space

import shapely



class PartialSensor(SensorTemplate):

    def __init__(self, space:Space):
        super().__init__()

        self.space = space
        self.data = []

        return None
    
    def sense(self, self_uav:UAV_v2):
        uav_list = self.space.get_uav_list()
        for uav in uav_list:
            # if uav intersects with self_uav, then add uav to data
            if shapely.intersects(uav.body, self_uav.body):
                self.data.append(uav)

