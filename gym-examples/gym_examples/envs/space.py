import math
import random
from typing import List
from uav_v2_template import UAV_v2_template
from shapely import Point



class Space:
    def __init__(self, max_uavs:int, max_vertiports:int, seed:int):
        # max_vertiports has to be EVEN 
        # max_uavs has to be ODD and one less than max_vertiports
        """
        Initializes the Space object with empty lists for UAVs and vertiports.
        number of vertiports has to be even.
        """
        self.max_vertiports = max_vertiports
        if max_vertiports % 2 != 0:
            self.max_vertiports = max_vertiports - 1

        if max_uavs <= max_vertiports:
            if max_uavs % 2 == 0:
                self.max_uavs = max_uavs - 1
            else: 
                self.max_uavs = max_uavs
        else:
            max_uavs = max_vertiports
            self.max_uavs = max_uavs - 1
            print(f'Max number of UAVs has to be less than or equal to the number of vertiports, setting max_uavs to {self.max_uavs}')

        self.uav_list:List = []
        self.vertiport_list:List = []
        self._seed = seed

    def set_vertiport(self,vertiport):
        """
        Adds a vertiport to the vertiport list.

        Args:
            vertiport: The vertiport to add.
        
        Returns:
            None
        """
        if len(self.vertiport_list) < self.max_vertiports:
            self.vertiport_list.append(vertiport)
        else:
            print('Max number of vertiports reached, additonal vertiports will not be added')
        return None
    
    def set_uav(self, uav:UAV_v2_template):
        """
        Adds a UAV to the UAV list.

        Args:
            uav (UAV_v2_template): The UAV to add.
        
        Returns:
            None
        """
        if len(self.uav_list) < self.max_uavs:
            self.uav_list.append(uav)
        else:
            print('Max number of UAVs reached, additonal UAVs will not be added')
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


    def create_circular_pattern_vertiports(self, number_of_vertiports, distance_between_vertiports) -> None:
        """
        Return coordinates evenly distributed in a circle, where consecutive coordinate pairs
        are equal distance away from one another and store them in space.vertiport_list.
        
        Args:
            distance (float): The distance between consecutive coordinates.
            
        Returns:
            None
        """
        if number_of_vertiports > self.max_vertiports:
            number_of_vertiports = self.max_vertiports
            print(f'Creating maximum number of vertiports, {number_of_vertiports}')
        
        if number_of_vertiports % 2 != 0:
            number_of_vertiports = number_of_vertiports - 1
            print(f'Creating even number of vertiports, {number_of_vertiports}')
        
        self.vertiport_pattern = 'circular'
        self.number_of_vertiports = number_of_vertiports
        self.distance_between_vertiports = distance_between_vertiports
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
    
    def create_random_pattern_vertiports(self, radius, number_of_vertiports) -> None:
        if number_of_vertiports > self.max_vertiports:
            number_of_vertiports = self.max_vertiports
            print(f'Creating maximum number of vertiports, {number_of_vertiports}')
        
        if number_of_vertiports % 2 != 0:
            number_of_vertiports = number_of_vertiports - 1
            print(f'Creating even number of vertiports, {number_of_vertiports}')

        self.vertiport_pattern = 'random'
        self.vertiport_circle_radius = radius
        self.number_of_vertiports = number_of_vertiports

        for i in range(self.number_of_vertiports):
            x = random.uniform(-self.vertiport_circle_radius, self.vertiport_circle_radius) #choose uniformly between -radius and radius
            y = random.uniform(-self.vertiport_circle_radius, self.vertiport_circle_radius) #choose uniformly between -radius and radius
            self.set_vertiport(Point(round(x,2), round(y,2)))
        
        return None


    def create_uavs(self, number_uavs, uav_type, has_agent ,controller, dynamics,sensor, radius, nmac_radius, detection_radius) -> None:
        # my space can have a max amount of agents
        # as UAVs are added, the space will keep track of the number of UAVs added 
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
        if number_uavs > self.max_uavs :
            number_uavs = self.max_uavs
            print(f'creating maximum number of UAVs, {number_uavs}')
        if number_uavs > self.number_of_vertiports:
            number_uavs = self.number_of_vertiports
            print(f'creating UAVs equal to the number of vertiports, {number_uavs}')
        
        if has_agent:
            number_uavs = number_uavs - 1
        self.number_of_uavs = number_uavs
        
        
        for _ in range(self.number_of_uavs):
            uav = uav_type(controller, dynamics, sensor, radius, nmac_radius, detection_radius)
            self.set_uav(uav)
        
        return None
        


    
    
    def assign_vertiports(self, assignment_type:str) -> None:
        """
        For a given space, assign start and end coordinates to UAVs.
        
        Args: 
            assignment_type (str): This string determines how the start-end points are assigned.
                                   Options are 'opposite', 'consecutive', 'random'.
                
        Returns:
            None  
        """
        print(f'Number of vertiports in space is {len(self.vertiport_list)}')
        print(f'Number of UAVs in space is {len(self.uav_list)}')
        
        if self.vertiport_pattern == 'circular':
            print(f'Vertiport pattern in space is {self.vertiport_pattern}')
        elif self.vertiport_pattern == 'random':
            print(f'Vertiport pattern in space is {self.vertiport_pattern}')
        else:
            print('No vertiport pattern set')
            raise ValueError('No vertiport pattern set')
        
        self.assignment_type = assignment_type
        coords_list_middle = int(len(self.vertiport_list)/2)
        coords_list_len = int(len(self.vertiport_list))

        
        local_veriport_list = self.vertiport_list.copy()
        local_uav_list = self.uav_list.copy()

        if self.vertiport_pattern == 'circular':
            if self.assignment_type == 'opposite':
                for i in range(len(self.uav_list)):
                    uav = self.uav_list[i]
                    start_idx = i % len(self.vertiport_list)
                    end_idx = (start_idx + coords_list_middle) % coords_list_len
                    start = self.vertiport_list[start_idx]
                    end = self.vertiport_list[end_idx]
                    uav.assign_start_end(start, end)
                
            elif self.assignment_type == 'consecutive':
                pass
            elif self.assignment_type == 'random':
                _start_list = self.vertiport_list.copy()
                _end_list = self.vertiport_list.copy()
                for i in range(len(self.uav_list)):
                    uav = self.uav_list[i]
                    start = random.choice(_start_list)
                    end = random.choice(_end_list)
                    uav.assign_start_end(start, end)
                    _start_list.remove(start)
                    _end_list.remove(end)
                    
            else:
                pass
        
        elif self.vertiport_pattern == 'random':
            random.seed(seed = self._seed)
            _start_list = self.vertiport_list.copy()
            _end_list = self.vertiport_list.copy()

            for i in range(len(self.uav_list)):
                uav = self.uav_list[i]
                start = random.choice(_start_list)
                end = random.choice(_end_list)
                uav.assign_start_end(start, end)
                _start_list.remove(start)
                _end_list.remove(end)


        return None
    
    #TODO: This method needs to be scrutinized 
    # def assign_vertiport_agent(self, agent:UAV_v2_template) -> None:
    #     """
    #     Assigns a start and end vertiport to the agent UAV.

    #     Args:
    #         agent (UAV_v2_template): The agent AutoUAV.
        
    #     Returns:
    #         None
    #     """
    #     agent.assign_start_end(self.vertiport_list[0], self.vertiport_list[-1])
    #     return None
    def assign_vertiport_agent(self, agent: UAV_v2_template) -> None:
        """
        Assigns a start and end vertiport to the agent UAV in a way that avoids
        conflicts with other UAVs.
        
        Args:
            agent (UAV_v2_template): The agent AutoUAV.
        
        Returns:
            None
        """
        coords_list_middle = int(len(self.vertiport_list)/2)
        coords_list_len = int(len(self.vertiport_list))
        
        # Instead of checking UAV start attributes, track assigned vertiports
        assigned_vertiports = set()
        for uav in self.uav_list:
            if hasattr(uav, 'start'):  # Check if start has been assigned
                assigned_vertiports.add(uav.start)
        
        # Find first unassigned vertiport
        for i in range(len(self.vertiport_list)):
            potential_start = self.vertiport_list[i]
            if potential_start not in assigned_vertiports:
                if self.vertiport_pattern == 'circular':
                    # Use the opposite point pattern
                    end_idx = (i + coords_list_middle) % coords_list_len
                    end = self.vertiport_list[end_idx]
                else:  # random pattern
                    # Choose from remaining vertiports
                    remaining = [v for v in self.vertiport_list 
                            if v != potential_start]
                    end = random.choice(remaining)
                    
                agent.assign_start_end(potential_start, end)
                return None
                
        raise RuntimeError("No available vertiports for agent - all are assigned")



