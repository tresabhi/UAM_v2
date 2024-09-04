import unittest
from shapely.geometry import Point
from geopandas import GeoSeries
from vertiport import Vertiport
from autonomous_uav import AutonomousUAV


class TestAutonomousUAV(unittest.TestCase):

    def setUp(self):
        """Set up the test environment."""
        self.start_vertiport = Vertiport(location=Point(0, 0))
        self.end_vertiport = Vertiport(location=Point(100, 100))
        self.uav = AutonomousUAV(
            start_vertiport=self.start_vertiport, end_vertiport=self.end_vertiport
        )

    def test_initialization(self):
        """Test the initialization of the AutonomousUAV object."""
        self.assertEqual(self.uav.start_vertiport, self.start_vertiport)
        self.assertEqual(self.uav.end_vertiport, self.end_vertiport)
        self.assertEqual(self.uav.landing_proximity, 50)
        self.assertEqual(self.uav.max_speed, 40)
        self.assertEqual(self.uav.current_speed, 0)

        # Test rendering properties
        self.assertEqual(self.uav.uav_footprint_color, "black")
        self.assertEqual(self.uav.uav_nmac_radius_color, "purple")
        self.assertEqual(self.uav.uav_detection_radius_color, "blue")

    def test_step(self):
        """Test the step function with given acceleration and heading correction."""
        initial_position = self.uav.current_position
        initial_speed = self.uav.current_speed
        acceleration = 1.0
        heading_correction = 10.0

        self.uav.step(acceleration, heading_correction)

        # After step, the UAV should have moved, so position should change
        self.assertNotEqual(self.uav.current_position, initial_position)
        self.assertEqual(self.uav.current_speed, initial_speed + acceleration)

        # The heading should have been updated
        self.assertNotEqual(
            self.uav.current_heading_deg,
            self.uav.angle_correction(self.uav.heading_deg),
        )


if __name__ == "__main__":
    unittest.main()
