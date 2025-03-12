import unittest
import geopandas as gpd
from osmnx import features as ox_features
from osmnx import geocode_to_gdf as geocode_to_gdf
from osmnx import projection as ox_projection
from shapely.geometry import Polygon

# Define the Airspace class
from airspace import Airspace

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