# Smart uav, will accept action from RL algorithm
from uav import UAV
from geopandas import GeoSeries
from vertiport import Vertiport


class AutonomousUAV(UAV):
    """Sub class of UAV, a smart UAV .
    It will accept acceleration and heading from RL algorithm"""

    def __init__(
        self,
        start_vertiport: Vertiport,
        end_vertiport: Vertiport,
        landing_proximity: float = 50,
        max_speed: float = 40,
    ) -> None:
        """Representation of UAV in airspace. UAV motion represented in 2D plane.
        Object is to move from start vertiport to end vertiport.
        A UAV instance requires a start and end vertiport.

        Args:
            start_vertiport (Vertiport): The starting vertiport for the autonomous UAV
            end_vertiport (Vertiport): The target vertiport for the autonomous UAV
            landing_proximity (float): How close the UAV needs to be to a vertiport to land
            max_speed (float): The maximum speed of the autonomous UAV
        """

        super().__init__(start_vertiport, end_vertiport, landing_proximity, max_speed)
        #! need to update the rendering properties,
        #! since i have called super how do i pass/update the attributes
        # UAV rendering-representation properties
        self.uav_footprint_color = "black"  # this color represents the UAV object
        self.uav_nmac_radius_color = "purple"
        self.uav_detection_radius_color = "blue"
        self.uav_collision_controller = None

    def step(self, acceleration: float, heading_correction: float) -> None:
        """Updates the UAV state with proper processing of actions"""
        
        # First update speed
        self._update_speed(acceleration)
        
        # Then update heading
        self._update_theta_d(heading_correction)
        
        # Then update position using new speed
        self._update_position(d_t=1)
        
        # Finally update reference
        self._update_ref_final_heading()
        
    # def step(self, acceleration: float, heading_correction: float) -> None:
    #     """
    #     Advances the UAV one time step

    #     Args:
    #         acceleration (float): The acceleration of the UAV
    #         heading_correction (float): The change in heading in degrees of the UAV
    #     """
    #     # First update speed
    #     self._update_speed(acceleration, d_t=1)

    #     # Then update heading
    #     self._update_theta_d(heading_correction)

    #     # Then update position using new speed
    #     self._update_position(d_t=1)
        
    #     # Finally update reference
    #     self._update_ref_final_heading()

    # def _update_speed(self, acceleration_from_controller: float, d_t: float = 1) -> None:
    #     """Update speed ensuring it stays non-negative"""
    #     updated_speed = self.current_speed + acceleration_from_controller * d_t
    #     self.current_speed = max(0.0, min(updated_speed, self.max_speed))  # Clamp between 0 and max_speed


# from vertiport import Vertiport
# from shapely import Point
# import numpy as np
# start = Vertiport(Point(0,0))
# end = Vertiport(Point(10,20))
# auto = Autonomous_UAV(start,end)

# print(np.array([auto.current_heading_deg - auto.current_ref_final_heading_deg]).shape)
