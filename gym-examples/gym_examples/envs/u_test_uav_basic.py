import unittest
from shapely.geometry import Point
from geopandas import GeoSeries
from vertiport import Vertiport
from das import CollisionController
from uav_basic import UAVBasic


class TestUAV(unittest.TestCase):

    def setUp(self):
        # Define start and end vertiports
        self.start_vertiport = Vertiport(location=Point(0, 0))
        self.end_vertiport = Vertiport(location=Point(1000, 1000))
        self.uav = UAVBasic(
            start_vertiport=self.start_vertiport, end_vertiport=self.end_vertiport
        )

    def test_initialization(self):
        # Test UAV initialization
        self.assertIsInstance(self.uav, UAVBasic)
        self.assertEqual(self.uav.start_point, self.start_vertiport.location)
        self.assertEqual(self.uav.end_point, self.end_vertiport.location)
        self.assertEqual(self.uav.current_position, self.start_vertiport.location)

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
        other_uav = UAVBasic(
            start_vertiport=self.start_vertiport, end_vertiport=self.end_vertiport
        )
        test = self.uav.current_position.distance(other_uav.current_position)
        distance = self.uav.get_intruder_distance(other_uav)
        self.assertEqual(distance, test)

    def test_get_intruder_speed(self):
        # Test intruder speed calculation
        other_uav = UAVBasic(
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
        other_uav = UAVBasic(
            start_vertiport=self.start_vertiport, end_vertiport=self.end_vertiport
        )
        other_uav.current_heading_deg = 1
        self.uav.current_heading_deg = 2
        relative_heading = self.uav.get_intruder_heading(other_uav)
        self.assertEqual(relative_heading, 1)

    #! This test requires a full sceen to have been rendered( test this function in a main test)
    # def test_get_state(self):
    #     # Test state retrieval
    #     self.uav_list = [self.uav]
    #     self.building_gdf = GeoSeries([Point(500, 500).buffer(100)])
    #     state = self.uav.get_state()
    #     self.assertIsInstance(state, tuple)
    #     self.assertEqual(len(state), 2)
    #! Test this function in a main test
    # def test_has_uav_collision(self):
    #     first_uav = UAVBasic(
    #         start_vertiport=self.start_vertiport, end_vertiport=self.end_vertiport
    #     )
    #     second_uav = UAVBasic(
    #         start_vertiport=Vertiport(location=Point(5, 0)),
    #         end_vertiport=self.end_vertiport,
    #     )
    #     list_UAV = list()
    #     list_UAV.append(first_uav)
    #     list_UAV.append(second_uav)
    #     print(f"List length: {len(list_UAV)}")
    #     print(f"First UAV: {list_UAV[0]}")
    #     print(f"Second UAV: {list_UAV[1]}")
    #     collision = self.uav.has_uav_collision(list_UAV)
    #     self.assertTrue(collision, bool(True))


#! This is missing get state and step tests and collision test

if __name__ == "__main__":
    unittest.main()
