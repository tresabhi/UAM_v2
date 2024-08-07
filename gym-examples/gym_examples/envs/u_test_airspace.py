import unittest
import geopandas as gpd
from osmnx import features as ox_features
from osmnx import geocode_to_gdf as geocode_to_gdf
from osmnx import projection as ox_projection
from shapely.geometry import Polygon

# Define the Airspace class (from the provided script)
from airspace import Airspace

# class Airspace:
#     def __init__(self, location_name: str, buffer_radius: float = 500) -> None:
#         """Airspace - Defines the location of the map. Imports key information on hospitals (no fly zone)

#         Args:
#             location_name (string): Location of the Airspace ie. "Austin, Texas, USA"
#             buffer_radius (int): distance around hospitals

#         Attributes:
#             location_name(string): Location of the Airspace ie. "Austin, Texas, USA"
#             buffer_radius (int): distance around hospitals
#             location_utm_gdf (gpd.GeoDataFrame): Location in UTM (Universal Transverse Mercator)
#             location_utm_hospital (ox_projection): location of hospital converted to UTM
#             location_utm_hospital_buffer (UTM): buffer around the hospital
#         """
#         self.location_name = location_name  # 'Austin, Texas, USA'
#         self.buffer_radius = buffer_radius

#         # location
#         location_gdf = geocode_to_gdf(
#             location_name
#         )  # converts named geocode - 'Austin,Texas' location to gdf
#         self.location_utm_gdf: gpd.GeoDataFrame = ox_projection.project_gdf(
#             location_gdf
#         )  # default projection - UTM projection
#         self.location_utm_gdf["boundary"] = (
#             self.location_utm_gdf.boundary
#         )  # adding column 'boundary'

#         # hospital
#         location_hospital = ox_features.features_from_polygon(
#             location_gdf["geometry"][0], tags={"building": "hospital"}
#         )
#         self.location_utm_hospital: gpd.GeoDataFrame = ox_projection.project_gdf(
#             location_hospital
#         )
#         self.location_utm_hospital_buffer = self.location_utm_hospital.buffer(
#             self.buffer_radius
#         )  # 500 meter buffer area

#     def __repr__(self) -> str:
#         return "Airspace({location_name})".format(location_name=self.location_name)


# # Test script for Airspace class


class TestAirspace(unittest.TestCase):
    def setUp(self):
        """Set up the test case with a known location."""
        self.airspace = Airspace("Austin, Texas, USA")

    def test_location_name(self):
        """Test that the location name is set correctly."""
        self.assertEqual(self.airspace.location_name, "Austin, Texas, USA")

    def test_buffer_radius(self):
        """Test that the buffer radius is set correctly."""
        self.assertEqual(self.airspace.buffer_radius, 500)

    def test_location_utm_gdf(self):
        """Test that the location UTM GeoDataFrame is not empty."""
        self.assertIsInstance(self.airspace.location_utm_gdf, gpd.GeoDataFrame)
        self.assertFalse(self.airspace.location_utm_gdf.empty)

    def test_location_utm_hospital(self):
        """Test that the hospital UTM GeoDataFrame is not empty."""
        self.assertIsInstance(self.airspace.location_utm_hospital, gpd.GeoDataFrame)
        self.assertFalse(self.airspace.location_utm_hospital.empty)

    def test_location_utm_hospital_buffer(self):
        """Test that the hospital buffer is created correctly."""
        self.assertFalse(self.airspace.location_utm_hospital_buffer.empty)
        self.assertTrue(
            all(
                isinstance(geom, Polygon)
                for geom in self.airspace.location_utm_hospital_buffer
            )
        )


if __name__ == "__main__":
    unittest.main()
