import numpy as np
import shapely
import pandas as pd
import geopandas as gpd
from geopandas import GeoSeries, GeoDataFrame
from osmnx import features as ox_features
from osmnx import geocode_to_gdf as geocode_to_gdf
from osmnx import projection as ox_projection
from typing import List, Tuple, Dict
import math
from shapely import Point
import random
from sklearn.cluster import KMeans as KM 

from vertiport import Vertiport
#FIX:
# this module will now handle creating objects in/on airspace
# vertiport creation
# restricted airspace creation
#  
class Airspace:
    def __init__(self, number_of_vertiports, location_name: str, buffer_radius: float = 500, airspace_tag_list=[], max_vertiports=25) -> None:   #! airspace feature has to be list of strings
        """Airspace - Defines the location of the map. Imports key information on hopitals(no fly zone)

        Args:
            location_name (string): Location of the Airspace ie. "Austin, Texas, USA"
            buffer_radius (int): distance around hospitals

        Attributes:
            location_name(string): Location of the Airspace ie. "Austin, Texas, USA"
            buffer_radius (int): distance around hospitals
            location_utm_gdf (gdp.GeoDataFrame): Location in UTM(Universal Transverse Mercator)
            location_utm_hospital ( ox_projection): location of hospital converted to UTM
            location_utm_hospital_buffer (UTM): buffer around the hospital
        """
        self.location_name = location_name  #'Austin, Texas, USA'
        self.buffer_radius = buffer_radius
        self.airspace_tag_list = airspace_tag_list

        # location - this is the airspace where we are working 
        location_gdf = geocode_to_gdf(location_name)  # converts named geocode - 'Austin,Texas' location to gdf
        self.location_utm_gdf: gpd.GeoDataFrame = ox_projection.project_gdf(location_gdf)  # default projection - UTM projection #! GeoDataFrame has deprication warning - need quick fix
        self.location_utm_gdf["boundary"] = (self.location_utm_gdf.boundary)  # adding column 'boundary'
        
        if self.airspace_tag_list:
            # airspace features and restricted airspace 
            self.location_tags = {}
            self.location_feature = {}
            self.location_utm = {}
            self.location_utm_buffer = {}
            
            self.airspace_restricted_area_buffer_array = []
            self.airspace_restricted_area_array = []
            
            for tag, tag_value in self.airspace_tag_list:
                self.location_tags[tag_value] = tag
                self.location_feature[tag_value] = ox_features.features_from_polygon(location_gdf["geometry"][0], tags={tag:tag_value})
                self.location_utm[tag_value] = ox_projection.project_gdf(self.location_feature[tag_value])
                self.location_utm_buffer[tag_value] = self.location_utm[tag_value].buffer(self.buffer_radius)
                self.airspace_restricted_area_array.append(self.location_utm[tag_value])
                self.airspace_restricted_area_buffer_array.append(self.location_utm_buffer[tag_value])
            
            self.restricted_airspace_buffer_geo_series = pd.concat(self.airspace_restricted_area_buffer_array)
            self.restricted_airspace_geo_series = pd.concat(self.airspace_restricted_area_array)


        # Vertiport
        self.number_of_vertiports = number_of_vertiports
        self.vertiport_list:List = []
        self.max_vertiports = max_vertiports 
        self.polygon_dict = {}

    def __repr__(self) -> str:
        return "Airspace({location_name})".format(location_name=self.location_name)

    def set_vertiport(self,vertiport):
        """
        Adds a vertiport to the vertiport list.

        Args:
            vertiport: The vertiport to add.
        
        Returns:
            None
        """
        if len(self.vertiport_list) < self.max_vertiports:
            self.vertiport_list.append(vertiport)
        else:
            print('Max number of vertiports reached, additonal vertiports will not be added')
        return None 
    
    def set_vertiports(self, vertiports:List[Vertiport], sample_number=None):
        '''Given a list of vertiports, 
            add randomly sampled 'sample_number' of vertiports 
            to airspace's vertiport_list'''
        
        if sample_number:
            sampled_vertiports = random.sample(vertiports, sample_number)
        
        for vertiport in sampled_vertiports:
            self.set_vertiport(vertiport)
        
        return None


    def get_vertiport_list(self):
        """
        Returns the list of vertiports.

        Returns:
            List: The list of vertiports.
        """
        return self.vertiport_list

    def create_n_random_vertiports(self, num_vertiports: int, seed = None) -> None:
        """
        Creates a specified number of random vertiports within the airspace.

        Args:
            num_vertiports (int): The number of vertiports to create.

        Returns:
            None

        Side Effects:
            - Creates the vertiports and updates the vertiports in the airspace list.
        """

        # Set seed if provided
        if seed is not None:
            print(f"Creating vertiports with seed: {seed}")
            random.seed(seed)
            np.random.seed(seed)

        if num_vertiports > self.number_of_vertiports:
            raise RuntimeError('Exceeds vertiport number defined for initialization')

        if self.airspace_tag_list:
            for tag_value in self.location_tags.keys():
                sample_space = self.location_utm_gdf.iloc[0,0].difference(
                    self.location_utm_buffer[tag_value].unary_union
                )
            sample_space_gdf = GeoSeries(sample_space)
        else: 
            sample_space = self.location_utm_gdf
            sample_space_gdf = sample_space.geometry

        
        sample_vertiport: GeoSeries = sample_space_gdf.sample_points(num_vertiports, rng=seed)#TODO: change seed to rng, to avoid warning 
        sample_vertiport_array: np.ndarray = shapely.get_parts(sample_vertiport[0])

        for location in sample_vertiport_array:
            self.vertiport_list.append(
                Vertiport(location=location, uav_list=[])
            )

        print(f"Created {len(self.vertiport_list)} vertiports with seed {seed}")

    def create_vertiport_at_location(self, position:Tuple)-> None:
        """Create a vertiport at position(x,y)."""
        position = Point(position[0], position[1])
        
        if self.airspace_tag_list:
            for tag_value in self.location_tags.keys():
                sample_space = self.location_utm_gdf.iloc[0,0].difference(
                    self.location_utm_buffer[tag_value].unary_union
                )
            sample_space_gdf = GeoSeries(sample_space)
        else: 
            sample_space = self.location_utm_gdf
            sample_space_gdf = sample_space.geometry


        sample_space_array: np.ndarray = shapely.get_parts(sample_space_gdf)

        for sample in sample_space_array:
            if sample.contains(position):
                print('Valid location for vertiport')
                _vertiport = Vertiport(position)
                return _vertiport
        
        print('Not a valid position for vertiport')

        return None
    


    def create_vertiport_from_polygon(self,polygon:shapely.Polygon) -> Vertiport:
        '''Given a polygon, find the centeroid of the polygon, 
        and place a vertiport at that polygon'''
        
        poly_centeroid = polygon.centroid
        return Vertiport(poly_centeroid)
        

    def create_vertiports_from_polygons(self,polygon_list:List[shapely.Polygon]) -> List[Vertiport]:
        '''Use polygons from polygon_list to create vertiports at each polygon'''
        
        vertiport_list = []
        for polygon in polygon_list:
            vertiport_list.append(self.create_vertiport_from_polygon(polygon))
        return vertiport_list
        

    def make_polygon_dict(self, tag_str):
        #TODO: check if tag_str in tag_list
        # if True, then use tag_str as key for dict
        
        '''Add polygons of specific "tag_str" to an instance dictionary called self.poly_dict.
        These polygons will be used to create vertiports using OSMNx tags'''


        self.polygon_dict[tag_str] = [obj for obj in self.location_utm[tag_str].geometry if isinstance(obj, shapely.Polygon)]

        return None
    

    def assign_region_to_vertiports(self, vertiport_list:List[Vertiport]) -> List[Vertiport]:
        #TODO: this needs to be an internal method 
        
        '''Assign regions to each vertioport from vertiport list. '''

        location_tuple = [(vertiport.x, vertiport.y) for vertiport in vertiport_list]

        kmeans = KM(n_clusters=4, random_state=0, n_init="auto").fit(location_tuple)
        
        print(f' These are the labels: {np.unique(kmeans.labels_)}')


        for i in range(len(kmeans.labels_)):
            vertiport = vertiport_list[i]
            vertiport.region = kmeans.labels_[i]

    
        return vertiport_list


    def make_region_dict(self, vertiport_list:List[Vertiport], num_regions:int) -> Dict:
        '''Return a dictionary, with keys as regions and values as list of vertiports of that region.
        This will be used later to sample vertiport from each region'''

        region_vertiport_dict = {}
        for region_id in range(num_regions):
            region_vertiport_dict[region_id] = []
            for vertiport in vertiport_list:
                if vertiport.region == region_id:
                    region_vertiport_dict[region_id].append(vertiport)
        

        return region_vertiport_dict
                    

    def sample_vertiport_from_region(self, region_dict:Dict, n_sample_from_region:int = 1):
        '''From the dictionary of regions with vertiports, 
        sample "n_sample_from_region" number of vertiports from vertiports list of that region'''

        sampled_vertiports = []
        
        for region in region_dict.keys():
            sampled_vertiports += random.sample(region_dict[region], n_sample_from_region)
        
        return sampled_vertiports


            






    
if __name__ == '__main__':
    airspace = Airspace(12, "Austin, Texas, USA", airspace_tag_list=[])
    vertiport = airspace.create_vertiport_at_location((630250,3358894))


    


