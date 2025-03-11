import math
import random
import numpy as np
from typing import List
from space import Space
from shapely import Point
from shapely.geometry import Polygon
from airspace import Airspace
from uav_v2_template import UAV_v2_template

class MapSpace(Space):
    """
    Extension of Space class that handles a real-world map with restricted airspace.
    Manages UAVs, vertiports, and their interactions within geographical constraints.
    """
    
    def __init__(self, max_uavs: int, max_vertiports: int, seed: int, airspace: Airspace):
        """
        Initialize the MapSpace.
        
        Args:
            max_uavs: Maximum number of UAVs allowed in the space
            max_vertiports: Maximum number of vertiports allowed in the space
            seed: Random seed for reproducibility
            airspace: Airspace object containing geographical data and restricted areas
        """
        super().__init__(max_uavs, max_vertiports, seed)
        self.airspace = airspace
        random.seed(seed)
        np.random.seed(seed)
    
    def create_random_vertiports(self, number_of_vertiports: int) -> None:
        """
        Create vertiports at random valid locations in the airspace.
        
        Args:
            number_of_vertiports: Number of vertiports to create
        """
        if number_of_vertiports > self.max_vertiports:
            number_of_vertiports = self.max_vertiports
            print(f'Creating maximum number of vertiports: {number_of_vertiports}')
        
        if number_of_vertiports % 2 != 0:
            number_of_vertiports = number_of_vertiports - 1
            print(f'Creating even number of vertiports: {number_of_vertiports}')
        
        self.vertiport_pattern = 'map_random'
        self.number_of_vertiports = number_of_vertiports
        
        # Get the airspace boundary
        airspace_boundary = self.airspace.location_utm_gdf.iloc[0, 0]
        
        # Get all restricted areas with buffers
        restricted_areas = []
        for tag_value in self.airspace.location_tags.keys():
            buffer_geo = self.airspace.location_utm_buffer[tag_value]
            for i in range(len(buffer_geo)):
                geometry = buffer_geo.iloc[i]
                if hasattr(geometry, 'geometry'):
                    restricted_areas.append(geometry.geometry)
                else:
                    restricted_areas.append(geometry)
        
        # Create vertiports
        vertiports_created = 0
        max_attempts = 100 * number_of_vertiports  # Avoid infinite loop
        attempts = 0
        
        while vertiports_created < number_of_vertiports and attempts < max_attempts:
            attempts += 1
            
            # Generate a random point within the airspace boundary
            if hasattr(airspace_boundary, 'bounds'):
                minx, miny, maxx, maxy = airspace_boundary.bounds
                x = random.uniform(minx, maxx)
                y = random.uniform(miny, maxy)
                point = Point(x, y)
                
                # Skip if point is not within airspace boundary
                if not point.within(airspace_boundary):
                    continue
                
                # Check if the point is not in any restricted area
                valid_location = True
                for area in restricted_areas:
                    try:
                        if point.within(area):
                            valid_location = False
                            break
                    except (TypeError, AttributeError):
                        # Skip problematic geometries
                        continue
                
                # Also check distance from existing vertiports
                min_distance = 500  # Minimum distance between vertiports in meters
                for existing_vertiport in self.vertiport_list:
                    if point.distance(existing_vertiport) < min_distance:
                        valid_location = False
                        break
                
                if valid_location:
                    self.set_vertiport(point)
                    vertiports_created += 1
        
        if vertiports_created < number_of_vertiports:
            print(f"Warning: Could only create {vertiports_created} of {number_of_vertiports} requested vertiports")
        
        return None
    
    def assign_vertiports(self, assignment_type: str) -> None:
        """
        Assign start and end vertiports to UAVs.
        
        Args:
            assignment_type: Strategy for assigning vertiports ('random', 'optimal', etc.)
        """
        print(f'Number of vertiports in space: {len(self.vertiport_list)}')
        print(f'Number of UAVs in space: {len(self.uav_list)}')
        
        if len(self.vertiport_list) < 2:
            raise ValueError("Need at least 2 vertiports to assign start and end points")
        
        self.assignment_type = assignment_type
        
        if assignment_type == 'random':
            # Each UAV gets a random pair of distinct vertiports
            available_starts = self.vertiport_list.copy()
            available_ends = self.vertiport_list.copy()
            
            for uav in self.uav_list:
                # Select random start vertiport
                start_vertiport = random.choice(available_starts)
                available_starts.remove(start_vertiport)
                
                # Create a temporary list excluding the start vertiport
                temp_ends = [v for v in available_ends if v != start_vertiport]
                
                # If no valid end vertiports, reuse one
                if not temp_ends:
                    temp_ends = [v for v in self.vertiport_list if v != start_vertiport]
                
                # Select random end vertiport
                end_vertiport = random.choice(temp_ends)
                if end_vertiport in available_ends:
                    available_ends.remove(end_vertiport)
                
                # Assign to UAV
                uav.assign_start_end(start_vertiport, end_vertiport)
                
                # Replenish available vertiports if necessary
                if not available_starts:
                    available_starts = self.vertiport_list.copy()
                if not available_ends:
                    available_ends = self.vertiport_list.copy()
        
        elif assignment_type == 'optimal':
            # Assign vertiports to minimize total distance or conflicts
            # This is a more complex assignment strategy
            pass
        
        else:
            # Default to random assignment
            self.assign_vertiports('random')
        
        return None
    
    def assign_vertiport_agent(self, agent: UAV_v2_template) -> None:
        """
        Assign start and end vertiports to the learning agent.
        
        Args:
            agent: The learning agent (Auto_UAV_v2)
        """
        if len(self.vertiport_list) < 2:
            raise ValueError("Need at least 2 vertiports to assign agent start and end points")
        
        # Find vertiports that aren't used as starting points by other UAVs
        used_starts = set()
        for uav in self.uav_list:
            if hasattr(uav, 'start'):
                used_starts.add(uav.start)
        
        available_starts = [v for v in self.vertiport_list if v not in used_starts]
        
        # If all vertiports are used, select one randomly
        if not available_starts:
            available_starts = self.vertiport_list
        
        # Select start vertiport
        start_vertiport = random.choice(available_starts)
        
        # Select end vertiport (different from start)
        available_ends = [v for v in self.vertiport_list if v != start_vertiport]
        end_vertiport = random.choice(available_ends)
        
        # Assign to agent
        agent.assign_start_end(start_vertiport, end_vertiport)
        print(f"Agent assigned start: ({start_vertiport.x}, {start_vertiport.y}), end: ({end_vertiport.x}, {end_vertiport.y})")
        return None
    
    def get_valid_position(self) -> Point:
        """
        Get a random valid position within the airspace (not in restricted areas).
        
        Returns:
            A valid Point object within the airspace
        """
        # Get the airspace boundary
        airspace_boundary = self.airspace.location_utm_gdf.iloc[0, 0]
        
        # Get all restricted areas with buffers
        restricted_areas = []
        for tag_value in self.airspace.location_tags.keys():
            buffer_geo = self.airspace.location_utm_buffer[tag_value]
            for i in range(len(buffer_geo)):
                geometry = buffer_geo.iloc[i]
                if hasattr(geometry, 'geometry'):
                    restricted_areas.append(geometry.geometry)
                else:
                    restricted_areas.append(geometry)
        
        # Find a valid position
        max_attempts = 1000
        for _ in range(max_attempts):
            # Generate a random point
            minx, miny, maxx, maxy = airspace_boundary.bounds
            x = random.uniform(minx, maxx)
            y = random.uniform(miny, maxy)
            point = Point(x, y)
            
            # Check if point is within airspace and not in restricted areas
            if point.within(airspace_boundary):
                valid = True
                for area in restricted_areas:
                    try:
                        if point.within(area):
                            valid = False
                            break
                    except (TypeError, AttributeError):
                        # Skip problematic geometries
                        continue
                
                if valid:
                    return point
        
        # Fallback if no valid position found
        print("Warning: Could not find a valid position in the airspace")
        # Return center of airspace as fallback
        centroid = airspace_boundary.centroid
        return Point(centroid.x, centroid.y)