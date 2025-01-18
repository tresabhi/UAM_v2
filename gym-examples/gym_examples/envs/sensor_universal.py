from sensor_template import SensorTemplate
from typing import List
from pandas import DataFrame
import numpy as np
import shapely
from uav_v2_template import UAV_v2_template
from utils import compute_time_to_impact

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
           
            if uav.id != self_uav.id:
                if self_uav.current_position.distance(uav.current_position) <= self.detection_radius:
                    data = {'other_uav id': uav.id,
                            'other_uav current_position':uav.current_position,
                            'other_uav current_speed':uav.current_speed,
                            'other_uav current_heading':uav.current_heading,
                            'other_uav radius': uav.radius}
                    self.data.append(data)
        return None
    

            # if uav.id != self_uav.id:
            #     relative_distance =  np.array([uav.current_position.x - self_uav.current_position.x, uav.current_position.y - self_uav.current_position.y])
            #     p_parallel_ego_frame = np.dot(relative_distance, self_uav_ref_prll)
            #     p_orthogonal_ego_frame = np.dot(relative_distance, self_uav_ref_orth)
            #     v_parallel_ego_frame = np.dot(np.array([uav.current_speed*np.cos(uav.current_heading), uav.current_speed*np.sin(uav.current_heading)]), self_uav_ref_prll)
            #     v_orthogonal_ego_frame = np.dot(np.array([uav.current_speed*np.cos(uav.current_heading), uav.current_speed*np.sin(uav.current_heading)]), self_uav_ref_orth)
            #     dist_between_agent_centers = shapely.distance(self_uav.current_position, uav.current_position)
            #     dist_to_other = round(dist_between_agent_centers - self_uav.radius - uav.radius, 2)
            #     combined_radius = self_uav.radius + uav.radius
            #     # relative_heading = self_uav_state['current_heading'] = uav_state['current_heading']
            #     time_to_impact = compute_time_to_impact(self_uav, uav)
            #     obs_array = np.array([p_parallel_ego_frame,
            #                           p_orthogonal_ego_frame,
            #                           v_parallel_ego_frame,
            #                           v_orthogonal_ego_frame,
            #                           uav.radius,
            #                           combined_radius,
            #                           dist_to_other,
            #                           time_to_impact])
            #     # save information in data
            #     self.data.append(obs_array)
        
    
    
    def get_data(self, sorting_criteria: str):
        """Return data of other UAVs in space.

        Args:
            sorting_criteria (str): Sorting method, one of ['closest first', 'closest last', 'time of impact'].

        Returns:
            np.ndarray: Sorted data with fixed shape (max_number_other_agents_observed, 7).
        """
        if sorting_criteria == 'closest first':
            sorted_data = sorted(self.data, key=lambda x: (x[6], x[1]))
        elif sorting_criteria == 'closest last':
            sorted_data = sorted(self.data, key=lambda x: (x[6], x[1]), reverse=True)
        elif sorting_criteria == 'time of impact':
            sorted_data = sorted(self.data, key=lambda x: (-x[7], -x[6], x[1]))
        else:
            raise RuntimeError('Not a valid sorting_criteria')

        # Convert to numpy array for consistent shape handling
        sorted_data = np.array(sorted_data)

        # Truncate or pad to ensure fixed shape
        if len(sorted_data) > self.max_number_other_agents_observed:
            # Truncate extra rows
            sorted_data = sorted_data[:self.max_number_other_agents_observed]
        elif len(sorted_data) < self.max_number_other_agents_observed:
            # Pad with zeros if there are fewer rows
            padding = np.zeros((self.max_number_other_agents_observed - len(sorted_data), 7))
            sorted_data = np.vstack((sorted_data, padding))

        return sorted_data
    

    def get_collision(self, self_uav:UAV_v2_template):
        uav_list = self.space.get_uav_list()
        
        for uav in uav_list:
            if uav.id == self_uav.id:
                continue
            else:
                if self_uav.current_position.buffer(self_uav.radius).intersects(uav.current_position.buffer(uav.radius)):
                    return True, (self_uav.id, uav.id)
        return False, None

    def get_nmac(self, self_uav:UAV_v2_template):
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
            