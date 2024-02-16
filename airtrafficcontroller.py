import numpy as np
from shapely import Point
import shapely
from geopandas import GeoSeries, GeoDataFrame
from typing import List
from airspace import Airspace
from uav import UAV
#from autonomous_uav import Autonomous_UAV


class ATC:
    def __init__(self, airspace:Airspace):
        self.airspace = airspace
        self.reg_uav_list = []
         
    
    
    def create_vertiport_at_location(self, position):
        '''Create a vertiport at position.
            Position is Point type, and 
            if position not within location, will return 0'''
        pass



    def create_n_random_vertiports(self, num_vertiports) -> np.ndarray:
        '''Create num_vertiports in airspace at random points'''
        sample_vertiport:GeoSeries = self.airspace.location_utm_gdf.sample_points(num_vertiports)
        sample_vertiport_array:np.ndarray = shapely.get_parts(sample_vertiport[0])
        return sample_vertiport_array

    def create_n_reg_uavs(self, num_uavs, vertiport_array):
        '''Create num_uavs instances of regualar uavs in airspace.
            This method creates the uavs and updates the reg_uav_list attribute'''
        
        for _ in range(num_uavs):
            start_end_array = np.random.choice(vertiport_array, 2)
            start, end = start_end_array[0], start_end_array[1]
            self.reg_uav_list.append(UAV(start, end))
         
    
    def set_start_end_uav(self, list_uav_airspace):
        '''Assign start-end point to all uavs in airspace'''
        pass


    def create_n_auto_uav(self,) -> List:
        return []

    def create_n_uavs(self, percent_auto):
        '''This method will create a mix of smart and regular uavs.
        The mix is controlled by percentage argument.'''
        pass
    