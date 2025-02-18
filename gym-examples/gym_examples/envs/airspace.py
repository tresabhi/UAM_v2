import pandas as pd
import geopandas as gpd
from osmnx import features as ox_features
from osmnx import geocode_to_gdf as geocode_to_gdf
from osmnx import projection as ox_projection


class Airspace:
    def __init__(self, location_name: str, buffer_radius: float = 500, airspace_tag_list=[]) -> None:   #! airspace feature has to be list of strings
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
        self.airspace_tag_list = airspace_tag_list #! add meaningful comment - use osmnx for definition of tag

        # location - this is the airspace where we are working 
        location_gdf = geocode_to_gdf(location_name)  # converts named geocode - 'Austin,Texas' location to gdf
        self.location_utm_gdf: gpd.GeoDataFrame = ox_projection.project_gdf(location_gdf)  # default projection - UTM projection #! GeoDataFrame has deprication warning - need quick fix
        self.location_utm_gdf["boundary"] = (self.location_utm_gdf.boundary)  # adding column 'boundary'
        
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


    def __repr__(self) -> str:
        return "Airspace({location_name})".format(location_name=self.location_name)



if __name__ == 'main':
    airspace = Airspace("Austin, Texas, USA", airspace_tag_list=[("building", "hospital"),("aeroway", "aerodrome")])
    print('Restricted airspace geoseries: ')
    print(airspace.restricted_airspace_geo_series)
    print('Restricted airspace geoseries-geometry: ')
    print(airspace.restricted_airspace_geo_series.geometry)