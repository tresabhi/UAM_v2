import unittest
from shapely.geometry import Point, Polygon
from geopandas import GeoDataFrame, GeoSeries
from airspace import Airspace
from vertiport import Vertiport
from uav_basic import UAVBasic
from autonomous_uav import AutonomousUAV
from atc import ATC


class TestATC(unittest.TestCase):

    def setUp(self):
        """Set up the test environment."""
        # Define the polygon for the airspace
        airspace_polygon = Polygon([(-100, -100), (-100, 100), (100, 100), (100, -100)])
        hospital_buffer = Polygon([(-10, -10), (-10, 10), (10, 10), (10, -10)])

        # Create GeoDataFrames for the airspace and hospital buffer
        location_utm_gdf = GeoDataFrame({"geometry": [airspace_polygon]})
        location_utm_hospital_buffer = GeoDataFrame({"geometry": [hospital_buffer]})

        self.airspace = Airspace(
            location_utm_gdf=location_utm_gdf,
            location_utm_hospital_buffer=location_utm_hospital_buffer,
        )
        self.atc = ATC(airspace=self.airspace)

    def test_create_n_random_vertiports(self):
        """Test creating random vertiports within the airspace."""
        num_vertiports = 5
        self.atc.create_n_random_vertiports(num_vertiports)
        self.assertEqual(len(self.atc.vertiports_in_airspace), num_vertiports)
        for vertiport in self.atc.vertiports_in_airspace:
            self.assertIsInstance(vertiport, Vertiport)

    def test_create_n_auto_uavs(self):
        """Test creating autonomous UAVs and assigning them random vertiports."""
        num_vertiports = 5
        self.atc.create_n_random_vertiports(num_vertiports)
        num_auto_uavs = 3
        self.atc.create_n_auto_uavs(num_auto_uavs)
        self.assertEqual(len(self.atc.auto_uavs_list), num_auto_uavs)
        for auto_uav in self.atc.auto_uavs_list:
            self.assertIsInstance(auto_uav, AutonomousUAV)
            self.assertIn(auto_uav.start_vertiport, self.atc.vertiports_in_airspace)
            self.assertIn(auto_uav.end_vertiport, self.atc.vertiports_in_airspace)
            self.assertNotEqual(auto_uav.start_vertiport, auto_uav.end_vertiport)

    def test_create_n_basic_uavs(self):
        """Test creating basic UAVs and assigning them random vertiports."""
        num_vertiports = 5
        self.atc.create_n_random_vertiports(num_vertiports)
        num_uavs = 3
        self.atc.create_n_basic_uavs(num_uavs)
        self.assertEqual(len(self.atc.basic_uav_list), num_uavs)
        for uav in self.atc.basic_uav_list:
            self.assertIsInstance(uav, UAVBasic)
            self.assertIn(uav.start_vertiport, self.atc.vertiports_in_airspace)
            self.assertIn(uav.end_vertiport, self.atc.vertiports_in_airspace)
            self.assertNotEqual(uav.start_vertiport, uav.end_vertiport)

    def test_has_reached_end_vertiport(self):
        """Test if a UAV has reached its end vertiport."""
        num_vertiports = 3
        self.atc.create_n_random_vertiports(num_vertiports)
        self.atc.create_n_auto_uavs(1)
        auto_uav = self.atc.auto_uavs_list[0]
        auto_uav.current_position = auto_uav.end_vertiport.location
        self.atc.has_reached_end_vertiport(auto_uav)
        self.assertIn(auto_uav, auto_uav.end_vertiport.uav_list)

    def test_has_left_start_vertiport(self):
        """Test if a UAV has left its start vertiport."""
        num_vertiports = 3
        self.atc.create_n_random_vertiports(num_vertiports)
        self.atc.create_n_auto_uavs(1)
        auto_uav = self.atc.auto_uavs_list[0]
        auto_uav.current_position = Point(200, 200)
        self.atc.has_left_start_vertiport(auto_uav)
        self.assertNotIn(auto_uav, auto_uav.start_vertiport.uav_list)

    def test_reassign_end_vertiport_of_uav(self):
        """Test reassigning the end vertiport of a UAV."""
        num_vertiports = 5
        self.atc.create_n_random_vertiports(num_vertiports)
        self.atc.create_n_basic_uavs(1)
        uav = self.atc.basic_uav_list[0]
        original_end_vertiport = uav.end_vertiport
        self.atc._reassign_end_vertiport_of_uav(uav)
        self.assertNotEqual(uav.end_vertiport, original_end_vertiport)
        self.assertIn(uav.end_vertiport, self.atc.vertiports_in_airspace)

    def test_vertiport_filtering(self):
        """Test filtering out a specific vertiport from the list."""
        num_vertiports = 5
        self.atc.create_n_random_vertiports(num_vertiports)
        some_vertiport = self.atc.vertiports_in_airspace[0]
        filtered_vertiports = self.atc._vertiport_filtering(some_vertiport)
        self.assertNotIn(some_vertiport, filtered_vertiports)
        self.assertEqual(len(filtered_vertiports), num_vertiports - 1)


if __name__ == "__main__":
    unittest.main()
