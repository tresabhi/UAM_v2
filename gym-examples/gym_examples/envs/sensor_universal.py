from sensor_template import SensorTemplate
from pandas import DataFrame
from uav_v2 import UAV_v2

class UniversalSensor(SensorTemplate):

    def __init__(self, space):
        super().__init__(space)


    def set_data(self, self_uav:UAV_v2) -> None:
        '''Collect data of other UAVs in space'''
        uav_list = self.space.get_uav_list()
        self_uav_state = self_uav.get_state()

        for uav in uav_list:
            # collect information about other UAV
            if uav.id != self_uav.id:
                uav_state = uav.get_state()
                relative_distance = self_uav_state['current_position'] - uav_state['current_position']
                relative_heading = self_uav_state['current_heading'] = uav_state['current_heading']
                other_uav_state = {'id':uav_state['id'],'start':uav_state['start'], 'end':uav_state['end'], 'relative_distance':relative_distance, 'relative_heading':relative_heading}
                # save information in data
                self.data.append(other_uav_state)
        return None
    
    def get_data(self):
        '''Return data of other UAVs in space'''
        return super().get_data()
    

    def get_collision(self, self_uav:UAV_v2):
        uav_list = self.space.get_uav_list()
        
        for uav in uav_list:
            if uav.id == self_uav.id:
                continue
            else:
                if self_uav.current_position.buffer(self_uav.radius).intersects(uav.current_position.buffer(uav.radius)):
                    return True, (self_uav.id, uav.id)
        return False, None

    def get_nmac(self, self_uav:UAV_v2):
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
            