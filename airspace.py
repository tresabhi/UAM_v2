import geopandas as gpd
from osmnx import features as ox_features
from osmnx import geocode_to_gdf as geocode_to_gdf 
from osmnx import projection as ox_projection



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
    