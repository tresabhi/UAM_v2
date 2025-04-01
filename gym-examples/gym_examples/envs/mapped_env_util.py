#mapped_env_util.py
from gymnasium.spaces import Dict, Box, Discrete
import numpy as np

def choose_obs_space_constructor(obs_space_string:str):
    '''This helper method uses the argument to return correct obs_space constructor for gym env
        Args:
            obs_space_string
        
        Returns:
            function (object)'''
    if obs_space_string == 'LSTM-A2C':
        return obs_space_seq
    elif obs_space_string == 'GNN-A2C':
        return obs_space_graph
    elif obs_space_string == 'UAM_UAV':
        return obs_space_uam







# Define sequential observation space for LSTM
def obs_space_seq(max_number_other_agents_observed):
    '''Gym observation space for LSTM-A2C model'''
    return Dict(
    {
        "no_other_agents": Box(
            low=0, high=max_number_other_agents_observed, shape=()
        ),
        "dist_goal": Box(low=0, high=10000, shape=(), dtype=np.float32),
        "heading_ego_frame": Box(low=-180, high=180, shape=(), dtype=np.float32),
        "current_speed": Box(low=0, high=50, shape=(), dtype=np.float32),
        "radius": Box(low=0, high=20, shape=(), dtype=np.float32),  # UAV size
        # Static object detection
        "static_collision_detected": Box(low=0, high=1, shape=(), dtype=np.int32),
        "distance_to_restricted": Box(low=0, high=10000, shape=(), dtype=np.float32),
        #FIX:  add another key:value for static object, the heading of static_obj
        
        # Other agent data
        "other_agent_state": Box(  # p_parall, p_orth, v_parall, v_orth, other_agent_radius, combined_radius, dist_2_other
            low=np.full(
                (max_number_other_agents_observed, 7), -np.inf
            ),
            high=np.full(
                (max_number_other_agents_observed, 7), np.inf
            ),
            shape=(
                max_number_other_agents_observed,
                7,
            ),
            dtype=np.float32,
        ),
    }
)

# Define graph observation space for GNN
def obs_space_graph(max_number_other_agents_observed):
    '''Gym obs space for GNN(and variants)-A2C model'''
    return Dict(
        {
            "num_other_agents": Box(low=0, high=100, shape=(), dtype=np.int64),
            "agent_dist_to_goal": Box(
                low=0, high=np.inf, shape=(), dtype=np.float32
            ),
            "agent_end_point": Box(
                low=np.array([-np.inf, -np.inf]),
                high=np.array([np.inf, np.inf]),
                shape=(2,),
                dtype=np.float32,
            ),
            "agent_current_position": Box(
                low=np.array([-np.inf, -np.inf]),
                high=np.array([np.inf, np.inf]),
                shape=(2,),
                dtype=np.float32,
            ),
            "static_collision_detected": Box(low=0, high=1, shape=(), dtype=np.int32),
            "distance_to_restricted": Box(low=0, high=10000, shape=(), dtype=np.float32),
            "graph_feat_matrix": Box(
                low=np.full(
                    (max_number_other_agents_observed + 1, 5), -np.inf
                ),
                high=np.full((max_number_other_agents_observed + 1, 5), np.inf),
                shape=(max_number_other_agents_observed + 1, 5),
                dtype=np.float32,
            ),
            "edge_index": Box(
                low=0,
                high=max_number_other_agents_observed,
                shape=(2, max_number_other_agents_observed),
                dtype=np.int64,
            ),
            "edge_attr": Box(
                low=0,
                high=np.inf,
                shape=(
                    max_number_other_agents_observed,
                    1,
                ),
                dtype=np.float32,
            ),
            "mask": Box(
                low=0,
                high=1,
                shape=(max_number_other_agents_observed + 1,),
                dtype=np.float32,
            ),
        }
    )



def obs_space_uam(auto_uav):
    '''Obs space for one intruder and restricted area'''
    return Dict(
            {
                # agent ID as integer
                "agent_id": Box(
                    low=0,
                    high=np.iinfo(np.int64).max,
                    shape=(1,),
                    dtype=np.int64,  #! find if it is possible to create ids that take less space
                ),
                # agent speed
                "agent_speed": Box(  #!need to rename velocity -> speed
                    low=-auto_uav.max_speed,  # agent's speed #! need to check why this is negative
                    high=auto_uav.max_speed,
                    shape=(1,),
                    dtype=np.float64,
                ),
                # agent deviation
                "agent_deviation": Box(
                    low=-360,
                    high=360,
                    shape=(1,),
                    dtype=np.float64,  # agent's heading deviation #!should this be -180 to 180, if yes then this needs to be corrected to -180 to 180
                ),
                # intruder detection
                "intruder_detected": Discrete(
                    2  # 0 for no intruder, 1 for intruder detected
                ),
                # intruder id
                "intruder_id": Box(
                    low=0,
                    high=np.iinfo(np.int64).max,
                    shape=(1,),
                    dtype=np.int64,  #! find if it is possible to create ids that take less space
                ),
                # distance to intruder
                "distance_to_intruder": Box(
                    low=0,
                    high=auto_uav.detection_radius,
                    shape=(1,),
                    dtype=np.float64,
                ),
                # Relative heading of intruder #!should this be corrected to -180 to 180,
                "relative_heading_intruder": Box(
                    low=-360, high=360, shape=(1,), dtype=np.float64
                ),
                "intruder_current_heading": Box(
                    low=-180, high=180, shape=(1,), dtype=np.float64
                ),  # Intruder's heading
                
                # restricted airspace
                "restricted_airspace_detected":Discrete(
                    2 
                ),
                # distance to airspace 
                "distance_to_restricted_airspace": Box(
                    low=0,
                    high=1000,
                    shape=(1,),
                    dtype=np.float64,
                ),
            }
        )


def get_obs_uam_uav(self):
    '''
    Internal method to collect observation for the agent/auto_uav
    Return: dict(observation dictionary)
    '''
    """
    Gets the observation space of the agent

    Args:
        agent_id (str): The ID of the target UAV

    Returns:
        obs (dict): The observation space of the agent
            agent_id
            agent_speed
            agent_deviation
            intruder_detected
            intruder_id
            distance_to_intruder
            relative_heading_intruder
            intruder_heading
    """

    #! this observation does not include the static objects
    #! add observation for static objects  
    agent_id = np.array([self.agent.id]) #
    agent_speed = self.agent.current_speed
    agent_deviation = self.agent.current_heading - self.agent.final_heading
    
    #FIX: self.agent.sensor.get_uav_detection(self.agent) -> could return None, when no intruder
    intruders_info = self.agent.sensor.get_uav_detection(self.agent)
    #FIX: define a way to choose intruder_index - closest or some other metric
    intruder_info = intruders_info[intruder_index] 

    if intruder_info:
        intruder_detected = 1
        intruder_id = np.array([intruder_info['other_uav_id']])
        intruder_pos = intruder_info['other_uav_current_position']
        intruder_heading = np.array([intruder_info['other_uav_current_heading']])
        distance_to_intruder = np.array(
            [self.agent.current_position.distance(intruder_info['other_uav_current_position'])]
        )
        relative_heading_intruder = np.array(
            [self.agent.current_heading - intruder_info['other_uav_current_heading']]
        )
    else:

        intruder_detected = 0
        intruder_id = np.array([0])
        distance_to_intruder = np.array([0])
        relative_heading_intruder = np.array([0])
        intruder_heading = np.array([0])

    #! restricted airspace detected - should this be detection/nmac
    # if the detection area of uav intersects with a building's detection area,
    # we should do the following -
    # 1) if intersection is detected for 'detection' argument ->
    #                   there should be a small penalty
    #                   based on distance between the uav_footprint and the actual building
    # 2) if there is no building detected the penalty should be zero
    # 3) just like intruder_detected in obs
    #    there will be restricted_airspace detected in obs
    # 4) restricted_airspace will have 0 for not detected 1 for detected,
    #    and if detected - distance will be added to the obs, if not detected distance is 0,
    #    this will be handled by the reward function collect obs and based on restricted_airspace (y/n)
    #    penatly is something or 0.

    #! use shapely method shapely.ops.nearest_points(geom1, geom2)->list; return a list of two points point_geom1, point_geom2
    
    if self.agent.sensor.get_ra_detection()[0] == True:
        ra_detected = 1
        ra_distance = self.agent.sensor.get_ra_detection()[1]['distance']
        ra_heading = self.agent.sensor.get_ra_detection()[1]['ra_heading'] 
    else:
        ra_detected = 0
        ra_distance = 0
        ra_heading = 0



    

    observation = {
        "agent_id": agent_id,
        "agent_speed": agent_speed,
        "agent_deviation": agent_deviation,
        "intruder_detected": intruder_detected,
        "intruder_id": intruder_id,
        "distance_to_intruder": distance_to_intruder,
        "relative_heading_intruder": relative_heading_intruder,
        "intruder_current_heading": intruder_heading,
        "ra_detected": ra_detected,
        "ra_distance": ra_distance,
        "ra_heading": ra_heading
    }

    return observation