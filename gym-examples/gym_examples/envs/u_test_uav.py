import unittest
from shapely.geometry import Point
from geopandas import GeoSeries
from vertiport import Vertiport
from das import CollisionController
from uav import UAV


class TestUAV(unittest.TestCase):

    def setUp(self):
        # Define start and end vertiports
        self.start_vertiport = Vertiport(location=Point(0, 0))
        self.end_vertiport = Vertiport(location=Point(1000, 1000))
        self.uav = UAV(
            start_vertiport=self.start_vertiport, end_vertiport=self.end_vertiport
        )

    def test_initialization(self):
        # Test UAV initialization
        self.assertIsInstance(self.uav, UAV)
        self.assertEqual(self.uav.start_point, self.start_vertiport.location)
        self.assertEqual(self.uav.end_point, self.end_vertiport.location)
        self.assertEqual(self.uav.current_position, self.start_vertiport.location)

    def test_step(self):
        # Test the step method
        initial_position = self.uav.current_position
        action = (1, 0)  # acceleration, heading_correction
        self.uav.step(action)
        self.uav.step(action)
        new_position = self.uav.current_position
        self.assertNotEqual(initial_position, new_position)

    def test_acceleration_controller(self):
        # Test the acceleration controller
        self.uav.current_speed = 0
        self.assertEqual(self.uav.acceleration_controller(), self.uav.max_acceleration)
        self.uav.current_speed = self.uav.max_speed
        self.assertEqual(self.uav.acceleration_controller(), self.uav.max_acceleration)

    def test_angle_correction(self):
        # Test the angle correction
        self.assertEqual(self.uav.angle_correction(360), 0)
        self.assertEqual(self.uav.angle_correction(-360), 0)
        self.assertEqual(self.uav.angle_correction(180), -180)
        self.assertEqual(self.uav.angle_correction(-180), -180)

    def test_get_intruder_distance(self):
        # Test intruder distance calculation
        other_uav = UAV(
            start_vertiport=self.start_vertiport, end_vertiport=self.end_vertiport
        )
        test = self.uav.current_position.distance(other_uav.current_position)
        distance = self.uav.get_intruder_distance(other_uav)
        self.assertEqual(distance, test)

    def test_get_intruder_speed(self):
        # Test intruder speed calculation
        other_uav = UAV(
            start_vertiport=self.start_vertiport, end_vertiport=self.end_vertiport
        )
        other_uav.current_speed = 20
        other_uav.current_heading_deg = 0
        self.uav.current_speed = 20
        self.uav.current_heading_deg = 0
        relative_speed = self.uav.get_intruder_speed(other_uav)
        self.assertEqual(relative_speed, 0)

    def test_get_intruder_heading(self):
        # Test intruder heading calculation
        other_uav = UAV(
            start_vertiport=self.start_vertiport, end_vertiport=self.end_vertiport
        )
        other_uav.current_heading_deg = 1
        self.uav.current_heading_deg = 2
        relative_heading = self.uav.get_intruder_heading(other_uav)
        self.assertEqual(relative_heading, 1)

    def test_get_state(self):
        # Test state retrieval
        uav_list = [self.uav]
        building_gdf = GeoSeries([Point(500, 500).buffer(100)])
        state = self.uav.get_state(uav_list, building_gdf)
        self.assertIsInstance(state, tuple)
        self.assertEqual(len(state), 2)


if __name__ == "__main__":
    unittest.main()
