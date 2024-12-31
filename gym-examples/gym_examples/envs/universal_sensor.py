from sensor_template import SensorTemplate
from pandas import DataFrame
from space import Space
from uav_v2 import UAV_v2

class UniversalSensor(SensorTemplate):

    def __init__(self):
        super().__init__()


    def set_data(self, space:Space, self_uav:UAV_v2) -> None:
        self.space = space
        uav_list = self.space.get_uav_list()
        self_uav_state = self_uav.get_state()

        for uav in uav_list:
            # collect information about other UAV
            if uav.id != self_uav.id:
                uav_state = uav.get_state()
                relative_distance = self_uav_state['current_position'] - uav_state['current_position']
                relative_heading = self_uav_state['current_heading'] = uav_state['current_heading']
                other_uav_state = {'id':uav_state['id'],'relative_distance':relative_distance, 'relative_heading':relative_heading}
                # save information in data
                self.data.append(other_uav_state)
        return None
    
    def get_data(self):
        return super().get_data()
