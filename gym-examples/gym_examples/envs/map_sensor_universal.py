from sensor_template import SensorTemplate
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import shapely
from uav_v2_template import UAV_v2_template
from utils import compute_time_to_impact

#FIX: delete this script


class UniversalSensor(SensorTemplate):

    def __init__(self, space):
        super().__init__(space)
        # Store reference to restricted airspace data (will be set later)
        self.restricted_airspace_geo_series = None
        self.restricted_airspace_buffer_geo_series = None

    def set_data(self, self_uav: UAV_v2_template) -> None:
        '''Collect data of other UAVs in space'''
        # Clear previous data
        self.data = []
        
        # get self_uav detection radius 
        self.detection_radius = self_uav.detection_radius
        
        # get UAV list from space
        uav_list: List[UAV_v2_template] = self.space.get_uav_list()

        for uav in uav_list:
            if uav.id != self_uav.id:
                if self_uav.current_position.distance(uav.current_position) <= self.detection_radius:
                    other_uav_data = {
                        'other_uav id': uav.id,
                        'other_uav_current_position': uav.current_position,
                        'other_uav_current_speed': uav.current_speed,
                        'other_uav_current_heading': uav.current_heading,
                        'other_uav_radius': uav.radius
                    }
                    self.data.append(other_uav_data)
        return None

    def get_data(self) -> List[Dict]:
        return self.data

    def get_collision(self, self_uav: UAV_v2_template) -> Tuple[bool, Tuple]:
        """Check for collisions with other UAVs"""
        uav_list = self.space.get_uav_list()
        
        for uav in uav_list:
            if uav.id == self_uav.id:
                continue
            else:
                if self_uav.current_position.buffer(self_uav.radius).intersects(uav.current_position.buffer(uav.radius)):
                    return True, (self_uav.id, uav.id)
                    
        return False, None

    def get_nmac(self, self_uav: UAV_v2_template) -> Tuple[bool, List]:
        """Check for Near Mid-Air Collisions (NMAC) with other UAVs"""
        nmac_list = []
        uav_list = self.space.get_uav_list()
        
        for uav in uav_list:
            if uav.id == self_uav.id:
                continue
            else:
                if self_uav.current_position.buffer(self_uav.nmac_radius).intersects(uav.current_position.buffer(uav.nmac_radius)):
                    nmac_list.append(uav)
                    
        if len(nmac_list) > 0:
            return True, nmac_list
            
        return False, nmac_list

    def set_restricted_airspace(self, restricted_areas, restricted_areas_buffer=None):
        """Set the restricted airspace data for collision detection"""
        self.restricted_airspace_geo_series = restricted_areas
        self.restricted_airspace_buffer_geo_series = restricted_areas_buffer

    def get_static_collision(self, self_uav: UAV_v2_template) -> Tuple[bool, str]:
        """Check if UAV has collided with restricted airspace
        
        Args:
            self_uav: The UAV to check for collisions
            
        Returns:
            Tuple[bool, str]: (collision_detected, collision_type)
        """
        if self.restricted_airspace_geo_series is None:
            return False, None
            
        # Create UAV footprint as a buffer around its current position
        uav_footprint = self_uav.current_position.buffer(self_uav.radius)
        
        # Check for intersection with restricted areas
        for idx, restricted_area in enumerate(self.restricted_airspace_geo_series.geometry):
            if uav_footprint.intersects(restricted_area):
                return True, f"restricted_area_{idx}"
                
        return False, None
        
    def get_static_nmac(self, self_uav: UAV_v2_template) -> Tuple[bool, str]:
        """Check if UAV is in near-miss situation with restricted airspace
        
        Args:
            self_uav: The UAV to check for NMAC
            
        Returns:
            Tuple[bool, str]: (nmac_detected, nmac_type)
        """
        if self.restricted_airspace_buffer_geo_series is None:
            return False, None
            
        # Create UAV NMAC zone as a buffer around its current position
        uav_nmac_zone = self_uav.current_position.buffer(self_uav.nmac_radius)
        
        # Check for intersection with restricted areas buffers
        for idx, restricted_buffer in enumerate(self.restricted_airspace_buffer_geo_series.geometry):
            if uav_nmac_zone.intersects(restricted_buffer):
                return True, f"restricted_buffer_{idx}"
                
        return False, None
        
    def get_static_detection(self, self_uav: UAV_v2_template) -> Tuple[bool, List]:
        """Check if any restricted airspace is within detection range
        
        Args:
            self_uav: The UAV to check for detections
            
        Returns:
            Tuple[bool, List]: (detection, list of detected static objects)
        """
        if self.restricted_airspace_buffer_geo_series is None:
            return False, []
            
        # Create UAV detection zone as a buffer around its current position
        uav_detection_zone = self_uav.current_position.buffer(self_uav.detection_radius)
        detected_objects = []
        
        # Check for intersection with restricted areas buffers
        for idx, restricted_buffer in enumerate(self.restricted_airspace_buffer_geo_series.geometry):
            if uav_detection_zone.intersects(restricted_buffer):
                # Calculate distance to the restricted area
                distance = self_uav.current_position.distance(restricted_buffer)
                detected_objects.append({
                    'object_id': f"restricted_buffer_{idx}",
                    'geometry': restricted_buffer,
                    'distance': distance
                })
                
        if detected_objects:
            # Sort by distance (closest first)
            detected_objects = sorted(detected_objects, key=lambda x: x['distance'])
            return True, detected_objects
            
        return False, []