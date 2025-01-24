# Smart uav, will accept action from RL algorithm
from uav import UAV
from geopandas import GeoSeries
from vertiport import Vertiport
import numpy as np

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

    # def step(self, acceleration: float, heading_correction: float) -> None:
    #     """Updates the UAV state with proper processing of actions"""
        
    #     # First update speed
    #     self._update_speed(acceleration)
        
    #     # Then update heading
    #     self._update_theta_d(heading_correction)
        
    #     # Then update position using new speed
    #     self._update_position(d_t=1)
        
    #     # Finally update reference
    #     self._update_ref_final_heading()

    def step(self, acceleration: float, heading_correction: float) -> None:
        """Enhanced step function with dynamic speed control"""
        # Get collision avoidance correction
        collision_correction = self._get_collision_avoidance_correction()
        
        # If not moving, ensure initial acceleration
        if self.current_speed < 0.1:
            acceleration = max(acceleration, 0.2)  # Maintain minimum acceleration when stopped
        else:
            # Dynamic speed control based on collision threat
            if collision_correction != 0:
                # Find closest intruder distance
                min_distance = min(self.current_position.distance(intruder.current_position) 
                                for intruder in self.intruder_uav_list)
                
                # Calculate desired speed based on proximity
                # At detection radius -> maintain speed
                # At NMAC radius -> reduce to 30% of max speed
                # Linear interpolation between these points
                distance_factor = (min_distance - self.nmac_radius) / (self.detection_radius - self.nmac_radius)
                distance_factor = np.clip(distance_factor, 0.0, 1.0)
                target_speed_factor = 0.3 + (0.7 * distance_factor)  # Varies from 0.3 to 1.0
                target_speed = self.max_speed * target_speed_factor
                
                # Adjust acceleration to achieve target speed smoothly
                if self.current_speed > target_speed:
                    # Gradual deceleration based on how far we are above target speed
                    speed_diff = self.current_speed - target_speed
                    acceleration = -min(self.max_acceleration, speed_diff)
                elif self.current_speed < target_speed:
                    # Allow acceleration if we're below target speed
                    acceleration = min(acceleration, (target_speed - self.current_speed))
        
        # Update speed with adjusted acceleration
        self._update_speed(acceleration)
        
        # Apply heading correction with priority to collision avoidance
        if collision_correction != 0:
            self._update_theta_d(collision_correction, collision_mode=True)
        else:
            self._update_theta_d(heading_correction, collision_mode=False)
        
        # Update position and reference heading
        self._update_position(d_t=1)
        self._update_ref_final_heading()

    def _get_collision_avoidance_correction(self) -> float:
        """Enhanced collision avoidance with immediate response"""
        if not hasattr(self, 'intruder_uav_list') or not self.intruder_uav_list:
            return 0.0

        immediate_correction = 0.0
        closest_distance = float('inf')
        most_urgent_threat = 0.0

        # Initial phase - handle immediate threats first
        for intruder in self.intruder_uav_list:
            rel_x = intruder.current_position.x - self.current_position.x
            rel_y = intruder.current_position.y - self.current_position.y
            distance = np.sqrt(rel_x**2 + rel_y**2)

            if distance >= self.detection_radius:
                continue

            # Determine relative motion
            rel_vx = (intruder.current_speed * np.cos(intruder.current_heading_radians) -
                    self.current_speed * np.cos(self.current_heading_radians))
            rel_vy = (intruder.current_speed * np.sin(intruder.current_heading_radians) -
                    self.current_speed * np.sin(self.current_heading_radians))
            
            # Calculate time to closest approach
            relative_velocity_sq = rel_vx**2 + rel_vy**2
            if relative_velocity_sq > 0:
                time_to_closest = max(0, -(rel_x*rel_vx + rel_y*rel_vy) / relative_velocity_sq)
            else:
                time_to_closest = 0

            # Calculate threat metrics
            distance_factor = 1.0 - (distance / self.detection_radius)
            urgency_factor = np.exp(-distance / (self.nmac_radius * 1.5))  # Stronger urgency response
            closing_speed = -(rel_x*rel_vx + rel_y*rel_vy) / distance if distance > 0 else 0
            collision_course_factor = max(0, -closing_speed / (self.max_speed * 2))

            # Immediate response for very close intruders
            if distance < self.nmac_radius * 2:
                threat_level = 1.0
                urgency_factor = 1.0
            else:
                threat_level = distance_factor * urgency_factor * (1 + collision_course_factor)

            if threat_level > most_urgent_threat:
                most_urgent_threat = threat_level
                closest_distance = distance

                # Calculate optimal avoidance direction
                to_intruder_angle = np.rad2deg(np.arctan2(rel_y, rel_x))
                rel_angle = ((to_intruder_angle - self.current_heading_deg + 180) % 360) - 180
                
                # Determine turn direction
                cross_product = rel_x * rel_vy - rel_y * rel_vx
                turn_direction = np.sign(cross_product) if abs(cross_product) > 1e-6 else np.sign(rel_angle)

                # Calculate base correction magnitude
                if distance < self.nmac_radius:
                    # Maximum correction for immediate threats
                    correction_magnitude = 90.0
                else:
                    # Progressive correction based on threat level
                    base_correction = 45.0 + (45.0 * urgency_factor)
                    correction_magnitude = base_correction * threat_level

                immediate_correction = turn_direction * correction_magnitude

                # Additional correction if head-on approach detected
                heading_diff = abs(((to_intruder_angle - self.current_heading_deg + 180) % 360) - 180)
                if heading_diff < 45 and closing_speed > 0:
                    immediate_correction *= 1.5  # Stronger correction for head-on scenarios

        return immediate_correction

    def _update_theta_d(self, heading_correction: float, collision_mode: bool = False, 
                    collision_correction: float = 0.0) -> None:
        """
        Enhanced heading control with clear priority between collision avoidance and goal seeking
        """
        # Calculate target heading to goal
        target_heading = np.rad2deg(np.arctan2(
            self.end_point.y - self.current_position.y,
            self.end_point.x - self.current_position.x
        ))
        
        # Calculate current heading error
        heading_diff = ((target_heading - self.current_heading_deg + 180) % 360) - 180
        
        if collision_mode:
            # Collision avoidance takes priority
            max_turn = 45.0  # Allow sharper turns for collision avoidance
            correction = np.clip(collision_correction, -max_turn, max_turn)
            self.current_heading_deg += correction
            
        else:
            # Normal goal-seeking behavior
            if abs(heading_diff) < 5.0:
                # Very close to desired heading, make minor adjustments
                correction = heading_diff * 0.5
            elif abs(heading_diff) < 30.0:
                # Moderate corrections when somewhat off course
                correction = np.sign(heading_diff) * min(abs(heading_diff) * 0.3, 15.0)
            else:
                # Larger corrections when significantly off course
                correction = np.sign(heading_diff) * 30.0
            
            # Apply heading correction from RL agent if it improves goal alignment
            if heading_correction != 0:
                proposed_correction = np.clip(heading_correction, -30.0, 30.0)
                proposed_heading = self.current_heading_deg + proposed_correction
                new_diff = ((target_heading - proposed_heading + 180) % 360) - 180
                
                if abs(new_diff) <= abs(heading_diff):
                    correction = proposed_correction
            
            self.current_heading_deg += correction
        
        # Normalize heading to [-180, 180]
        self.current_heading_deg = ((self.current_heading_deg + 180) % 360) - 180
        self.current_heading_radians = np.deg2rad(self.current_heading_deg)

# from vertiport import Vertiport
# from shapely import Point
# import numpy as np
# start = Vertiport(Point(0,0))
# end = Vertiport(Point(10,20))
# auto = Autonomous_UAV(start,end)

# print(np.array([auto.current_heading_deg - auto.current_ref_final_heading_deg]).shape)
