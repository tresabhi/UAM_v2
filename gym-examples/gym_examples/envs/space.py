import math
from typing import List
from uav_v2_template import UAV_v2_template
from shapely import Point



class Space:
    def __init__(self, number_of_vertiports:int, distance_between_vertiports:float, assignment_type:str):
        """
        Initializes the Space object with empty lists for UAVs and vertiports.
        number of vertiports has to be even.
        """
        if number_of_vertiports % 2 != 0:
            raise RuntimeError('number_of_vertiports argument requires an even number.')
        
        self.number_of_vertiports = number_of_vertiports
        self.distance_between_vertiports = distance_between_vertiports
        self.assignment_type = assignment_type
        self.uav_list:List = []
        self.vertiport_list:List = []

    def set_vertiport(self,vertiport):
        """
        Adds a vertiport to the vertiport list.

        Args:
            vertiport: The vertiport to add.
        
        Returns:
            None
        """
        self.vertiport_list.append(vertiport)
        return None
    
    def set_uav(self, uav:UAV_v2_template):
        """
        Adds a UAV to the UAV list.

        Args:
            uav (UAV_v2_template): The UAV to add.
        
        Returns:
            None
        """
        self.uav_list.append(uav)
        return None

    def get_vertiport_list(self):
        """
        Returns the list of vertiports.

        Returns:
            List: The list of vertiports.
        """
        return self.vertiport_list
    
    def get_uav_list(self) -> List[UAV_v2_template]:
        """
        Returns the list of UAVs.

        Returns:
            List[UAV_v2_template]: The list of UAVs.
        """
        return self.uav_list
    


    def remove_uavs_by_id(self, ids_to_remove):
        """
        Removes UAV objects from the list based on their id attribute.
        Used when a collision is detected between two UAVs.

        Args:
            ids_to_remove (set): A set of IDs to remove from the list.

        Returns:
            None
        """
        self.uav_list = [uav for uav in self.uav_list if uav.id not in ids_to_remove]

        return None


    def create_vertiports(self) -> None:
        """
        Return coordinates evenly distributed in a circle, where consecutive coordinate pairs
        are equal distance away from one another and store them in space.vertiport_list.
        
        Args:
            distance (float): The distance between consecutive coordinates.
            
        Returns:
            None
        """
        
        # Angle between consecutive points in the polygon
        angle_between_points = (2 * math.pi) / self.number_of_vertiports
        
        # Radius of the circle to ensure the given distance
        radius = self.distance_between_vertiports / (2 * math.sin(math.pi / self.number_of_vertiports))
        

        for i in range(self.number_of_vertiports):
            angle = i * angle_between_points
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            self.set_vertiport(Point(round(x,2), round(y,2)))
        
        return None


    def assign_vertiports(self):
        """
        For a given space, assign start and end coordinates to UAVs.
        
        Args: 
            assignment_type (str): This string determines how the start-end points are assigned.
                                   Options are 'opposite', 'consecutive', 'random'.
                
        Returns:
            None  
        """
        coords_list_middle = int(len(self.vertiport_list)/2)
        coords_list_len = int(len(self.vertiport_list))
        
        local_veriport_list = self.vertiport_list.copy()


        if self.assignment_type == 'opposite':
            for i in range(len(local_veriport_list)):
                uav = self.uav_list[i]
                start = local_veriport_list[i]
                end = local_veriport_list[(i+coords_list_middle)%coords_list_len]
                uav.assign_start_end(start, end)
                local_veriport_list.remove(start)
                # this will not work - will need to use some sort of pointer, else the logic for choosing vertiport may not work - test this to see if it works 
            
        elif self.assignment_type == 'consecutive':
            pass
        elif self.assignment_type == 'random':
            pass
        else:
            pass


        return None



    def create_uav(self, n, uav_type, controller, dynamics,sensor, radius, nmac_radius) -> None:
        """
        Create "n" quantity of UAVs.

        Args:
            n (int): Number of UAVs to create.
            uav_type: The type of UAV to create.
            controller: The controller for the UAV.
            dynamics: The dynamics of the UAV.
            sensor: The sensor for the UAV.
            radius: The radius of the UAV.
            nmac_radius: The NMAC radius of the UAV.
        
        Returns:
            None
        """

        # n has to be less than or equal to the number of vertiports and has to be even number 
        if n > self.number_of_vertiports or n % 2 != 0:
            raise RuntimeError('n argument requires an even number and less than or equal to the number of vertiports.')
        


        for _ in range(n):
            uav = uav_type(controller, dynamics, sensor, radius, nmac_radius)
            self.set_uav(uav)
        return None
