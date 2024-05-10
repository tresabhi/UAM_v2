    # collision_avoidance_controller_basic.py
    # 

from geopandas import GeoSeries
import numpy as np 


class Collision_controller:
    
    def uav_collision_detection(self, uav, uav_list): #TODO - uav_list argument includes self, thats why I am observing 'Collision Detected' continuously
        '''
        This procedure has to be performed first 
        if distance of UAV from vertiport less than equal to some value
            DO NOT PERFORM collision avoidance 
        THEN - i can perform what I have below
        '''
        uav_own = uav
        modified_uav_list = []
        for ith_uav in uav_list:
            if uav_own.id != ith_uav.id:
                modified_uav_list.append(ith_uav)
        
        uav_footprint_polygon = GeoSeries(uav_own.current_position).buffer(uav_own.uav_footprint).iloc[0]
        for uav_other in modified_uav_list:
            if uav_footprint_polygon.intersects(GeoSeries(uav_other.current_position).buffer(uav_other.uav_footprint).iloc[0]):
                print('UAV Collision Detected')


    def uav_nmac_detection(self, uav, uav_list): #TODO - uav_list argument includes self, thats why I am observing 'Collision Detected' continuously
        '''
        This procedure has to be performed first 
        if distance of UAV from vertiport less than equal to some value
            DO NOT PERFORM collision avoidance 
        THEN - i can perform what I have below
        '''
        uav_own = uav
        modified_uav_list = []
        for ith_uav in uav_list:
            if uav_own.id != ith_uav.id:
                modified_uav_list.append(ith_uav)
        
        uav_self = GeoSeries(uav_own.current_position).buffer(uav_own.nmac_radius).iloc[0]
        for uav_other in modified_uav_list:
            if uav_self.intersects(GeoSeries(uav_other.current_position).buffer(uav_other.nmac_radius).iloc[0]):
                #TODO - this is not a good implementation - needs a good fix 
                uav_self.current_heading_deg = np.random.randint(low=-45, high=46)
                uav_self.current_heading_radians = np.deg2rad(uav_self.current_heading_deg)
        

    def static_collision_detection(self, uav, static_object_df:GeoSeries):
        '''
        check intersection with uav list - here return is true or false, true meaning intersection  
        check intersection with raz_list
        '''
        uav_footprint_polygon = GeoSeries(uav.current_position).buffer(uav.uav_footprint).iloc[0]

        for i in range(len(static_object_df)):
            if uav_footprint_polygon.intersects(static_object_df.geometry.iloc[i]):
                print('Static object collision')

    def static_nmac_detection(self, uav, static_object_df) : #static_object_df -> dataframe  # return contact_uav_id 
        '''
         check intersection with uav list -  return is geoseries with true or false, true meaning intersection with contact_uav 
         collect contact_uav id for true in geoseries
         use the contact_uav id to collect information of the uav - 
         required info 
                        contact_uav - heading, distance from contact_uav(can be calculated using position), velocity
                        ownship_uav     - deviation, velocity, has_intruder
                        relative bearing - calculate as -> ownship_heading - absolute_angle_between_contact_and_ownship
        
         check intersection with static_object_df ??  
        '''

        uav_self = GeoSeries(uav.current_position).buffer(uav.nmac_radius).iloc[0]
        # print('type: ', type(uav_self.iloc[0]))

        for i in range(len(static_object_df)):
            if uav_self.intersects(static_object_df.iloc[i]):
                # 90 degree clockwise rotation 
                #TODO - need to update this NOW !!!
                uav.current_heading_deg = uav.current_ref_final_heading_deg - 45
                uav.current_heading_radians = np.deg2rad(uav.current_heading_deg)
