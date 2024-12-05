# Deterministic uavs
from random import sample as random_sample
import numpy as np
from shapely.geometry import Point
from geopandas import GeoSeries
from vertiport import Vertiport
from das import CollisionController
# TODO - abstract controller, basic collision controller
# from collision_avoidance_controller_basic import uav_collision_detection, uav_nmac_detection, static_collision_detection, static_nmac_detection


class UAV:
    '''Representation of UAV in airspace. UAV motion represented in 2D plane. 
     Object is to move from start vertiport to end vertiport.
     A UAV instance requires a start and end vertiport.
     '''

    def __init__(self,
                 start_vertiport: Vertiport,  
                 end_vertiport: Vertiport,
                 landing_proximity: float = 50., 
                 max_speed: float = 43,
                 ) -> None:
        """
        Args:
            start_vertiport (Vertiport): The starting vertiport for the UAV
            end_vertiport (Vertiport): The target vertiport for the UAV
            landing_proximity (float): How close the UAV needs to be to a vertiport to land
            max_speed (float): The maximum speed of the UAV
        """

        # UAV builtin properties
        self.heading_deg = np.random.randint(-178,178) + np.random.rand() # random heading between -180 and 180
        self.collision_radius = 17 #H175 nose to tail length of 17m,
        self.uav_footprint = 17
        self.nmac_radius = 150 #NMAC radius
        self.detection_radius = 550
        self.uav_footprint_color = 'blue' # this color represents the UAV object 
        self.uav_nmac_radius_color = 'orange'
        self.uav_detection_radius_color = 'green'
        self.uav_collision_controller = None

        # UAV technical properties
        self.id = id(self)
        self.current_speed = 0
        self.max_speed:float = max_speed
        self.max_acceleration = 1 # m/s^2, this has been obtained from internet 
        self.landing_proximity = landing_proximity # UAV object considered landed when within this radius

        # UAV soft properties
        self.leaving_start_vertiport = False
        self.reaching_end_vertiport = False

        # Vertiport assignement
        self.start_vertiport = start_vertiport
        self.end_vertiport = end_vertiport

        # UAV position properties
        self.start_point:Point = self.start_vertiport.location
        self.end_point:Point = self.end_vertiport.location
        self.current_position:Point = self.start_point

        # UAV heading properties
        self.current_heading_deg:float = self.angle_correction(self.heading_deg)
        self.current_heading_radians = np.deg2rad(self.current_heading_deg)

        # Final heading calculation
        self.current_ref_final_heading_rad = np.arctan2(self.end_point.y - self.current_position.y, 
                                                                    self.end_point.x - self.current_position.x)
        self.current_ref_final_heading_deg = self.angle_correction(np.rad2deg(self.current_ref_final_heading_rad))

    def acceleration_controller(
        self,
    ) -> float:
        """
        Determines the acceleration of the UAV based on current speed and distance to the end point
        """

        if self.current_speed == 0:
            acc = self.max_acceleration
        elif self.current_position.distance(self.end_point) <= 500:
            acc = -((self.max_speed) ** 2 / (2 * 500))
        elif self.current_speed <= self.max_speed:
            acc = self.max_acceleration
        else:
            acc = 0

        return acc

    def angle_correction(self, deg: float) -> float:
        normalized_deg = (deg + 180) % 360 - 180
        return normalized_deg

    def get_collision(self, uav_list: list["UAV"]) -> bool:
        """Return True if collision has been detected."""

        intruder_uav_list = self.get_intruder_uav_list(uav_list, "collision")

        if len(intruder_uav_list) != 0:
            return True
        else:
            return False

    def get_intruder_distance(self, other_uav: "UAV") -> float:
        return self.current_position.distance(other_uav.current_position)

    def get_intruder_speed(self, other_uav: "UAV") -> float:
        rel_heading = self.get_intruder_heading(other_uav)
        return self.current_speed - (
            np.cos(np.deg2rad(rel_heading)) * other_uav.current_speed
        )

    def get_intruder_heading(self, other_uav: "UAV") -> float:
        return self.current_heading_deg - other_uav.current_heading_deg

    def get_intruder_uav_list(
        self, uav_list: list["UAV"], radius_str: str
    ) -> list["UAV"]:
        """
        Here the self.intruder_uav_list is created everytime as an empty list,
        So everystep this attribute is an empty list and its populated with uavs that are within any(detection, nmac, collision) radius.
        There is no return from this method, the data is stored in the attribute and should be accessed immediately after calling this method.
        Any subsequent routines can call the attribute and use the attribute for data processing
        """
        if radius_str == "detection":
            own_radius = other_radius = self.detection_radius

        elif radius_str == "nmac":
            own_radius = other_radius = self.nmac_radius

        elif radius_str == "collision":
            own_radius = other_radius = self.collision_radius

        else:
            raise RuntimeError("Unknown radius string passed.")

        self.intruder_uav_list = []
        other_uav_list = self.get_other_uav_list(uav_list)

        for other_uav in other_uav_list:
            if self.uav_polygon(own_radius).intersects(
                other_uav.uav_polygon(other_radius)
            ):
                self.intruder_uav_list.append(other_uav)

        return self.intruder_uav_list

    def get_other_uav_list(self, uav_list: list["UAV"]) -> list["UAV"]:
        other_uav_list = []
        for uav in uav_list:
            if uav.id != self.id:
                other_uav_list.append(uav)
        return other_uav_list

    def get_uav_current_heading_arrow(self):
        x, y = self.current_position.x, self.current_position.y
        r = self.detection_radius
        dx = r * np.cos(self.current_heading_radians)
        dy = r * np.sin(self.current_heading_radians)
        return x, y, dx, dy

    def get_uav_final_heading_arrow(self):
        x, y = self.current_position.x, self.current_position.y
        r = self.detection_radius
        dx = r * np.cos(self.current_ref_final_heading_rad)
        dy = r * np.sin(self.current_ref_final_heading_rad)
        return x, y, dx, dy

    def get_state(
        self,
        uav_list: list["UAV"],
        building_gdf: GeoSeries,
        radius_str: str = "detection",
    ) -> tuple[tuple[bool, dict], None | dict]:
        static_state = self.get_state_static_obj(building_gdf, radius_str)
        dynamic_state = self.get_state_dynamic_obj(uav_list, radius_str)

        return static_state, dynamic_state

    def get_state_dynamic_obj(self, uav_list: list["UAV"], radius_str: str = 'detection') -> None | dict:
        '''
        Get state of UAV based on radius string argument.
        Return intruder UAVs info that have been detected. 
        Return None if no UAV is detected.
        '''

        intruder_uav_list = self.get_intruder_uav_list(uav_list,radius_str)

        if len(intruder_uav_list) == 0:
            return None
        else:
            '''
            If there are more than one intruder, set first in the list as current intruder,
            and iterate through the list to find the intruder that is nearest. 
            State information, will be built using the nearest intruder. 
            '''
            current_intruder:UAV = intruder_uav_list[0] #set first uav in intruder_list as current intruder 
            # sort based on distance
            for ith_intruder in intruder_uav_list:
                if self.get_intruder_distance(ith_intruder) < self.get_intruder_distance(current_intruder):
                    current_intruder = ith_intruder #nearest intruder is current intruder

            intruder_state_info = {'own_id':self.id,
                                'own_pos': self.current_position,
                                'own_current_heading':self.current_heading_deg,
                                'intruder_id': current_intruder.id,
                                'intruder_pos': current_intruder.current_position,
                                'intruder_current_heading': current_intruder.current_heading_deg,
                                }

            return intruder_state_info

    def get_state_static_obj(self, building_gdf: GeoSeries, radius_str: str = 'detection') -> tuple[bool, dict]:

        '''Return (true/false, current_heading) based on intersection with any building '''

        if radius_str == 'detection':
            own_radius = self.detection_radius
        elif radius_str == 'nmac':
            own_radius  = self.nmac_radius
        elif radius_str == 'collision':
            own_radius = self.collision_radius
        else:
            raise RuntimeError('Unknown radius string passed.')

        building_polygon_count = len(building_gdf)
        intersection_list = []

        for i in range(building_polygon_count):
            '''Loop through all buildings in the 
            "building_gdf" that has been passed in as method's argument '''
            intersection_list.append(self.uav_polygon(own_radius).intersection(building_gdf.iloc[i]))

        intersection_with_building = any(intersection_list)
        own_state_info = self.current_heading_deg

        return intersection_with_building , own_state_info

    def refresh_uav(
        self,
    ) -> None:
        """Update the start vertiport of a UAV
        to a new start vertiport,
        argument of this method"""
        self.leaving_start_vertiport = False
        self.reaching_end_vertiport = False

    def uav_polygon(self, dimension) -> GeoSeries:
        return GeoSeries(self.current_position).buffer(dimension).iloc[0]

    def uav_polygon_plot(self, dimension) -> GeoSeries:
        return GeoSeries(self.current_position).buffer(dimension)

    def update_end_point(self,) -> None:
        '''Updates the UAV end point using its own end_vertiport location'''
        self.end_point = self.end_vertiport.location

    def update_start_point(self,) -> None:
        self.start_point = self.start_vertiport.location

    def _update_position(self,d_t: float = 1,) -> None:
        '''Internal method. Updates current_position of the UAV after d_t seconds.
           This uses a first order Euler's method to update the position.
           '''
        update_x = self.current_position.x + self.current_speed * np.cos(self.current_heading_radians) * d_t 
        update_y = self.current_position.y + self.current_speed * np.sin(self.current_heading_radians) * d_t 
        self.current_position = Point(update_x,update_y)

    def _update_ref_final_heading(
        self,
    ) -> None:
        """Internal method. Updates the heading of the aircraft, pointed towards end_point"""
        self.current_ref_final_heading_rad = np.arctan2(
            self.end_point.y - self.current_position.y,
            self.end_point.x - self.current_position.x,
        )
        self.current_ref_final_heading_deg = np.rad2deg(
            self.current_ref_final_heading_rad
        )

    def step(self, action: tuple[float, float]) -> Point:
        """Updates the UAV state with proper processing of actions"""
        acceleration, heading_correction = action
        
        # First update speed
        self._update_speed(acceleration)
        
        # Then update heading
        self._update_theta_d(heading_correction)
        
        # Then update position using new speed
        self._update_position(d_t=1)
        
        # Finally update reference
        self._update_ref_final_heading()
        
        return self.current_position

    def _update_speed(self, acceleration: float, d_t: float = 1) -> None:
        """Simplified speed update with direct acceleration application"""
        # Apply acceleration directly
        new_speed = self.current_speed + acceleration * d_t
        # Clamp to valid range
        self.current_speed = max(0.0, min(new_speed, self.max_speed))

    def _update_theta_d(self, heading_correction: float = 0) -> None:
        """
        Update heading with strong goal-seeking behavior and stability
        """
        # Calculate target heading to goal
        target_heading = np.rad2deg(np.arctan2(
            self.end_point.y - self.current_position.y,
            self.end_point.x - self.current_position.x
        ))
        
        # Calculate heading difference using smallest angle
        heading_diff = ((target_heading - self.current_heading_deg + 180) % 360) - 180
        
        if heading_correction == 0:
            # Autonomous goal-seeking behavior
            if abs(heading_diff) < 1.0:  # Very close to target heading
                # Maintain current heading
                return
            elif abs(heading_diff) < 10.0:  # Close to target
                turn_rate = np.sign(heading_diff) * min(abs(heading_diff) * 0.1, 5.0)
            else:
                # Standard turn rate for larger corrections
                turn_rate = np.sign(heading_diff) * min(20.0, abs(heading_diff) * 0.2)
                
            self.current_heading_deg += turn_rate
        else:
            # Only apply external correction if it improves goal alignment
            new_heading = self.current_heading_deg + heading_correction
            new_diff = ((target_heading - new_heading + 180) % 360) - 180
            
            if abs(new_diff) <= abs(heading_diff):
                self.current_heading_deg = new_heading
        
        # Normalize heading
        self.current_heading_deg = ((self.current_heading_deg + 180) % 360) - 180
        self.current_heading_radians = np.deg2rad(self.current_heading_deg)

    def undef_state(self, uav_list: list["UAV"]) -> dict:
        # deviation = self.current_heading_deg - self.current_ref_final_heading_deg
        # speed = self.current_speed
        # num_intruder = self.calculate_intruder(uav_list)
        # intruder_distance = self.calculate_intruder_distance()
        # intruder_speed = self.calculate_intruder_speed()
        # intruder_heading = self.calculate_intruder_heading()

        # state = {'deviation':deviation,
        #               'speed':speed,
        #               'num_intruder':num_intruder,
        #               'intruder_distance':intruder_distance,
        #               'intruder_speed':intruder_speed,
        #               'intruder_heading':intruder_heading}
        # return state
        pass

    def get_global_state(self, uav_list: list["UAV"]):
        pass
