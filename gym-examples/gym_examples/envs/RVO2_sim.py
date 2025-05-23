# import modules 
import rvo2
from uav_v2_template import UAV_v2_template
from controller_template import ControllerTemplate
import shapely
from typing import List, Dict 
from map_env_revised import MapEnv




# connect this file with all ORCA agents 
class RVO2_simulator:
    def __init__(self, timestep,radius,max_speed, mapped_env_orca_agent_list:List[UAV_v2_template] = None):
        # create the ORCA sim 
        self.rvo2_sim = rvo2.PyRVOSimulator(# the arguments should be added to __init__ above
                                        timestep = timestep, #sim timestep 
                                        neighborDist = 1.5, # these numbers are from example.py - ./Python-RVO2
                                        maxNeighbor = 5, #  
                                        timeHorizon = 1.5, #
                                        timeHorizonConst = 2, #
                                        radius = radius, # size of UAV
                                        maxSpeed = max_speed, # max speed of UAV
                                        )
        
        self.orca_polygon_list = []
        # ORCA agent list from MAPPED_ENV
        self.mapped_env_orca_agent_list = mapped_env_orca_agent_list 
        # RVO2_SIM - orca agent dict
        #                        key        :     value
        #                        str(uav_id), RVO2_agent_obj
        self.orca_agent_dict:Dict = {}
        # MAPPING from RVO2 ORCA agent to MAPPED_ENV AGENT
        self.orca_agent_to_mapped_agent:Dict[str, UAV_v2_template] = {}


    

    def step(self):
        '''This method needs to talk with MappedEnv, 
           Internally updates the position of UAV to MappedEnv'''
        # step moves all agents in ORCA/RVO2
        self.rvo2_sim.doStep()

        #assign prefVelocity to all agents 
        for orca_agent_key, orca_agent_value in self.orca_agent_dict.items():
            
            mapped_agent = self.orca_agent_to_mapped_agent[orca_agent_key]
            orca_agent_current_position = rvo2.getAgentPosition(self.orca_agent_dict[orca_agent_key])
            
            # updating the MAPPED_ENV UAV agent's current_position
            self.orca_agent_to_mapped_agent[orca_agent_key].current_position = orca_agent_current_position
            
            #
            self.rvo2_sim.setAgentPrefVelocity(orca_agent_value, self.getPrefVelocity(mapped_agent))

        # get agent current position and update orca_agent_dict


        return None


    def reset(self,):
        '''Collect information from MappedEnv
           There is no information that needs to go back from this method to MappedEnv'''
        
        
        # ADD agents to RVO2_SIM
        for mapped_env_orca_agent in self.mapped_env_orca_agent_list:
            
            (x,y) = mapped_env_orca_agent.start.x, mapped_env_orca_agent.start.y 
            
            # key string - using mapped_agent.id 
            uav_id_str = str(mapped_env_orca_agent.id)
            # mapping agent uav_id_str to mapped_agent
            self.orca_agent_to_mapped_agent[uav_id_str] = mapped_env_orca_agent
            # adding agent to PyORCA/RVO2 sim
            self.orca_agent_dict[uav_id_str] = self.rvo2_sim.addAgent((x,y))
        

        # ADD OBSTACLE IN RVO2_SIM - restriced airspace polygons 
        for orca_polygon in self.orca_polygon_list:
            self.rvo2_sim.addObstacle(orca_polygon)
            self.rvo2_sim.processObstacles()
        
        # ASSIGN prefVelocity IN RVO2_SIM - to agents
        #   orca_agent:str
        for orca_agent_key, orca_agent_value in self.orca_agent_dict.items():
            mapped_agent = self.orca_agent_to_mapped_agent[orca_agent_key]
            self.rvo2_sim.setAgentPrefVelocity(orca_agent_value, self.getPrefVelocity(mapped_agent)) 
        
        return None
        
    def getPrefVelocity(self, agent:UAV_v2_template):
        '''Return the prefVelocity tuple for RVO2 sim'''
        
        current_position = agent.current_position
        goal = agent.end

        goalDirection = goal - current_position
        norm_goalDirection = goalDirection/shapely.length(goal, current_position) #need to define magnitude
        prefVelocity = norm_goalDirection * agent.max_speed
        prefVelocityTuple = (prefVelocity.x, prefVelocity.y)
        return prefVelocityTuple #prefVelocity has to be a tuple
    
    def set_polygon_coords(self, poly_list):
        '''Call this method in Mapped_env's reset 
        ASSIGN argument - mapped_env.restricted_airspace_buffer_geo_series'''

        # poly-list is [poly1, poly2, poly3, poly4, poly5, ...... poly_n]
        # poly_n = [[x1 y1], [x2 y2], ....[xn yn]]
        
        for i in range(len(poly_list)):
            airspace_polygon = poly_list[i]
            orca_polygon = []
            for coords in shapely.get_coordinates(airspace_polygon):
                tuple_coords = (coords[0], coords[1])
                orca_polygon.append(tuple_coords)
            self.orca_polygon_list.append(orca_polygon)
        
        return None




