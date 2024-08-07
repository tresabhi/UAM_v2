import unittest
import numpy as np
from shapely.geometry import Point

# Define the CollisionController class (from the provided script)
from das import CollisionController

# class CollisionController:
#     """
#     Controllers are defined for individual UAVs,
#     A controller will be used for a certain type of UAV,
#     The controller will accept the observation of one UAV
#     and based on that specific UAV's observation, it will assign action to that specific UAV.
#     """

#     def __init__(self) -> None:
#         self.name = "Baseline Collision Controller"

#     def get_action(self, state: list) -> tuple[float, float]:
#         """
#         Gets the action for a basic UAV

#         Args:
#             state (list): The state space for the UAV
#                 ?all UAVs position, speed, current_heading, ref_final_heading

#         Returns:
#             acceleration (int): the acceleration of the UAV in m/s^2 [0 or -1]
#             heading_correction (int): the heading correction of the UAV in degrees [0, 25 or -25]
#         """
#         if state[0][0] is False and state[1] is None:
#             acceleration = 0
#             heading_correction = 0

#         elif state[0][0] is False and isinstance(state[1], dict):
#             own_pos = state[1]["own_pos"]
#             int_pos = state[1]["intruder_pos"]
#             own_heading = state[1]["own_current_heading"]
#             int_heading = state[1]["intruder_current_heading"]

#             del_x = int_pos.x - own_pos.x
#             del_y = int_pos.y - own_pos.y
#             own_quadrant = self.get_quadrant(own_heading)
#             intruder_quadrant = self.get_quadrant(int_heading)
#             if (del_x > 0) and (del_y > 0):
#                 if (own_quadrant == 1) and (intruder_quadrant == 4):
#                     heading_correction = 25
#                     acceleration = -1
#                 else:
#                     heading_correction = 0
#                     acceleration = 0
#             elif (del_x < 0) and (del_y > 0):
#                 if (own_quadrant == 2) and (intruder_quadrant == 3):
#                     heading_correction = -25
#                     acceleration = -1
#                 else:
#                     heading_correction = 0
#                     acceleration = 0
#             elif (del_x < 0) and (del_y < 0):
#                 if (own_quadrant == 4) and (intruder_quadrant == 1):
#                     heading_correction = 25
#                     acceleration = -1
#                 else:
#                     heading_correction = 0
#                     acceleration = 0
#             elif (del_x > 0) and (del_y < 0):
#                 if (own_quadrant == 3) and (intruder_quadrant == 2):
#                     heading_correction = -25
#                     acceleration = -1
#                 else:
#                     heading_correction = 0
#                     acceleration = 0
#             else:
#                 raise RuntimeError("Action not from scenario")

#         elif state[0][0] is True and state[1] is None:
#             acceleration = 0
#             current_heading = state[0][1]

#             if current_heading < 0:
#                 current_heading += 360
#             elif current_heading > 180:
#                 current_heading -= 360

#             if 0 <= current_heading or current_heading <= 180:
#                 heading_correction = 25
#             elif -180 <= current_heading or current_heading <= 0:
#                 heading_correction = -25
#             else:
#                 raise RuntimeError(
#                     f"DAS module - state[0][0] is True and state[1] is None, current heading {state[0][1]}"
#                 )

#         elif state[0][0] is True and isinstance(state[1], dict):
#             acceleration = 0
#             heading_correction = 0

#         else:
#             raise RuntimeError(
#                 "DAS module: static and dynamic states do not match the conditionals"
#             )

#         return acceleration, heading_correction

#     def get_quadrant(self, theta: float) -> int:
#         """
#         Gets the quadrant based on the angle theta provided.
#         There are 4 quadrants.
#         2|1
#         ---
#         4|3

#         Args:
#             Theta (float): angle relative to the x-axis (ccw is +)

#         Returns:
#             (int): Only 1-4
#         """
#         if (theta >= 0) and (theta < 90):
#             return 1
#         elif (theta >= 90) and (theta <= 180):
#             return 2
#         elif (theta < 0) and (theta >= -90):
#             return 3
#         elif (theta >= -180) and (theta < -90):
#             return 4
#         else:
#             raise RuntimeError("DAS Error: Invalid heading")


# Test script for CollisionController class


class TestCollisionController(unittest.TestCase):
    def setUp(self):
        """Set up the test case with a CollisionController instance."""
        self.controller = CollisionController()

    def test_no_static_no_dynamic(self):
        """Test case with no static and no dynamic objects."""
        state = [(False, None), None]
        acceleration, heading_correction = self.controller.get_action(state)
        self.assertEqual(acceleration, 0)
        self.assertEqual(heading_correction, 0)

    def test_static_no_dynamic(self):
        """Test case with static objects but no dynamic objects."""
        state = [(True, 45), None]
        acceleration, heading_correction = self.controller.get_action(state)
        self.assertEqual(acceleration, 0)
        self.assertEqual(heading_correction, 25)

    def test_dynamic_no_static(self):
        """Test case with dynamic objects but no static objects."""
        state = [
            (False, None),
            {
                "own_pos": Point(0, 0),
                "intruder_pos": Point(10, 10),
                "own_current_heading": 45,
                "intruder_current_heading": -135,
            },
        ]
        acceleration, heading_correction = self.controller.get_action(state)
        self.assertEqual(acceleration, -1)
        self.assertEqual(heading_correction, 25)

    def test_static_and_dynamic(self):
        """Test case with both static and dynamic objects."""
        state = [
            (True, 45),
            {
                "own_pos": Point(0, 0),
                "intruder_pos": Point(10, 10),
                "own_current_heading": 45,
                "intruder_current_heading": -135,
            },
        ]
        acceleration, heading_correction = self.controller.get_action(state)
        self.assertEqual(acceleration, 0)
        self.assertEqual(heading_correction, 0)

    def test_quadrant(self):
        """Test the get_quadrant method."""
        self.assertEqual(self.controller.get_quadrant(45), 1)
        self.assertEqual(self.controller.get_quadrant(135), 2)
        self.assertEqual(self.controller.get_quadrant(-45), 3)
        self.assertEqual(self.controller.get_quadrant(-135), 4)


if __name__ == "__main__":
    unittest.main()
