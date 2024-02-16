import geopandas as gpd
from osmnx import features as ox_features
from osmnx import geocode_to_gdf as geocode_to_gdf 
from osmnx import projection as ox_projection
import numpy as np
from typing import List, Tuple
from shapely.geometry import Point


class Airspace:
    def __init__(self, location_name:str):

        self.location_name = location_name#'Austin, Texas, USA'
        #location
        location_gdf = geocode_to_gdf(location_name) #converts named geocode - 'Austin,Texas' location to gdf 
        self.location_utm_gdf:gpd.GeoDataFrame = ox_projection.project_gdf(location_gdf) #default projection - UTM projection 
        self.location_utm_gdf['boundary'] = self.location_utm_gdf.boundary #adding column 'boundary' 
        #hospital        
        location_hospital = ox_features.features_from_polygon(location_gdf['geometry'][0], tags={'building':'hospital'})
        self.location_utm_hospital = ox_projection.project_gdf(location_hospital)
        self.location_utm_hospital_buffer = self.location_utm_hospital.buffer(500)
        
        #self._location_property -> (private property) lists the properties of the location, which is a gpd.GeoDataFrame
    
    def __repr__(self) -> str:
        return ('Airspace({location_name})'.format(location_name = self.location_name))
    
    #TODO - this function will be superseded by methods from airtrafficcontroller 
    def get_random_start_end_points(self) -> Tuple[Point, Point]:
        '''Returns two random points as a tuple of Point object from the location map.'''
        sample_points_gdf = self.location_utm_gdf.sample_points(2) # returns two random points in the map (GeoSeries)
        sample_points = sample_points_gdf[0] # returns a MultiPoint object
        start_point = sample_points.geoms[0]
        end_point = sample_points.geoms[1]
        return start_point, end_point
    
    #TODO - this function will be superseded by methods from airtrafficcontroller 
    def set_start_end_point(start, end):
        '''User defined start and end point for uav'''
        return (Point(start), Point(end))
    
    #TODO - this function will be superseded by methods from airtrafficcontroller 
    def create_random_vertiports(self, num_vertiports):
        sample_points_gdf = self.location_utm.sample_points(6)
        sample_points = sample_points_gdf[0]


    


# airspace = Airspace(location_name='Austin, Texas, USA')
# print(airspace.get_random_start_end_points())
# print(airspace.location_utm_gdf.crs)
