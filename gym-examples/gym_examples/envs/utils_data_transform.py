from typing import List, Dict
import numpy as np
from utils import compute_time_to_impact
import math
from gymnasium.spaces import Dict, Box, Discrete


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


#### OBS SPACE CONSTRUCTOR ####
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

# Define uam observation space 
def obs_space_uam(auto_uav):
    '''Obs space for one intruder and restricted area'''
    return Dict(
            {
                # agent ID for debugging
                "agent_id": Box(
                    low=0,
                    high=np.iinfo(np.int64).max,
                    shape=(1,),
                    dtype=np.int64,
                ),
                # agent speed
                "agent_speed": Box(  #!TODO ensure that attribute is being properly referenced 
                    low=-auto_uav.max_speed,
                    high=auto_uav.max_speed,
                    shape=(1,),
                    dtype=np.float64,
                ),
                "agent_current_heading": Box(
                    low=-180,
                    high=180,
                    shape=(),
                    dtype=np.float32,
                    ),
                # agent deviation
                "agent_deviation": Box(
                    low=-180,
                    high=180,
                    shape=(1,),
                    dtype=np.float64,
                ),
                # intruder detection
                "intruder_detected": Discrete(
                    2  # 0 for no intruder, 1 for intruder detected
                ),
                # intruder id for debugging
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
                    low=-180, 
                    high=180, 
                    shape=(1,), 
                    dtype=np.float64
                ),
                # intruder's current heading
                "intruder_current_heading": Box(
                    low=-180, 
                    high=180, 
                    shape=(1,), 
                    dtype=np.float64,
                ),
                # restricted airspace
                "restricted_airspace_detected":Discrete(
                    2 # 0 for no ra, 1 for ra detected
                ),
                # distance to airspace 
                "distance_to_restricted_airspace": Box(
                    low=0,
                    high=10000,
                    shape=(1,),
                    dtype=np.float64,
                ),
                "relative_heading_restricted_airspace": Box(
                    low=-180,
                    high=180,
                    shape=(1,),
                    dtype=np.float64,
                )
            }
        )

def obs_space_uam():
    return Dict(
        {
            "agent_dist_to_goal": Box(
                low=0, 
                high=10000, 
                shape=(), 
                dtype=np.float32
            ),
            "agent_speed": Box(
                low=0.0,
                high=50.0,
                shape=(),
                dtype=np.float32,
            ),
            "agent_current_heading": Box(
                low=-180,
                high=180,
                shape=(),
                dtype=np.float32,
            ),
            # "agent_deviation": Box(
            #     low=-180,
            #     high=180,
            #     shape=(),
            #     dtype=np.float32,
            # ),
            "intruder_detected": Box(
                low=0, 
                high=1, 
                shape=(), 
                dtype=np.int32,
            ),
            "distance_to_intruder": Box(
                low=0.0,
                high=10000.0,
                shape=(),
                dtype=np.float32,
            ),
            "relative_heading_intruder": Box(
                low=-180,
                high=180,
                shape=(),
                dtype=np.float32,
            ),
            "intruder_current_heading": Box(
                low=-180,
                high=180,
                shape=(),
                dtype=np.float32,
            ),
            "restricted_airspace_detected": Box(
                low=0, 
                high=1, 
                shape=(), 
                dtype=np.int32,
            ),
            "distance_to_restricted_airspace": Box(
                low=0,
                high=10000.0,
                shape=(),
                dtype=np.float32,
            ),
            "relative_heading_restricted_airspace": Box(
                low=-180,
                high=180,
                shape=(),
                dtype=np.float32,
            )
        }
    )



#### DATA TRANSFORMATION ####
def transform_sensor_data(data,
                          max_number_other_agents_observed,
                          return_data_format:str,
                          sorting_criteria=None):
    """Return data of other UAVs in space.
    Args:
        sorting_criteria (str): Sorting method, one of ['closest first', 'closest last', 'time of impact'].
    Returns:
        np.ndarray: Sorted data with fixed shape (max_number_other_agents_observed, 7).
    """
    if return_data_format == 'LSTM-A2C':
        transformed_data = transform_for_sequence(data, sorting_criteria, max_number_other_agents_observed)
        return transformed_data
    elif return_data_format == 'GNN-A2C':
        transformed_data = transform_for_graph(data, max_number_other_agents_observed)
        return transformed_data
    elif return_data_format == 'UAM_UAV':
        transformed_data = transform_for_uam(data)
        return transformed_data
    else:
        raise RuntimeError('Incorrect data_format passed to transform_sensor_data')
    

def transform_for_sequence(data: List[Dict], sorting_criteria: str, max_number_other_agents_observed: int) -> Dict:
    """
    Transform UAV data into a fixed-size array with relative states for each UAV, with padding and masking.
    Args:
        data (List[Dict]): List of dictionaries containing UAV information.
        max_number_other_agents_observed (int): Maximum number of agents to consider.
        sorting_criteria (str): Criteria for sorting other agents.
    Returns:
        Dict: Transformed UAV data, including a mask for valid entries and the number of observed agents.
    """
    transformed_data = {}
    # Auto UAV aka Host data
    host_data = data[0]
    # Create numpy array from Point coordinates correctly
    host_position = np.array([host_data['current_position'].x,
                              host_data['current_position'].y])
    # Create numpy array from current speed and current heading correctly
    host_velocity = np.array([host_data['current_speed'] * np.cos(host_data['current_heading']),
                            host_data['current_speed'] * np.sin(host_data['current_heading'])])
    host_heading = host_data['current_heading']
    host_radius = host_data['radius']
    host_ref_prll = host_data['ref_prll']
    host_ref_orth = host_data['ref_orth']
    # Initialize empty array for other UAVs' states
    other_uav_states = np.zeros((max_number_other_agents_observed, 7), dtype=np.float32)
    num_other_agents = 0
    if len(data) > 1:
        other_uav_data = data[1:]  # Skip host agent
        uav_state_list = []
        # Process other UAVs
        for other_uav in other_uav_data:
            # Fix: Create position as a numpy array with both x and y components
            other_position = np.array([
                other_uav['other_uav_current_position'].x,
                other_uav['other_uav_current_position'].y
            ])
            # Fix: Create velocity as a numpy array with both x and y components
            other_velocity = np.array([
                other_uav['other_uav_current_speed'] * np.cos(other_uav['other_uav_current_heading']),
                other_uav['other_uav_current_speed'] * np.sin(other_uav['other_uav_current_heading'])
            ])
            other_radius = other_uav['other_uav_radius']
            # Relative position and velocity
            rel_pos = other_position - host_position
            p_parallel_ego_frame = np.dot(rel_pos, host_ref_prll)
            p_orthog_ego_frame = np.dot(rel_pos, host_ref_orth)
            dist_between_centers = np.linalg.norm(rel_pos)
            dist_to_other = dist_between_centers - host_radius - other_radius
            # Velocity projection
            v_parallel_ego_frame = np.dot(other_velocity, host_ref_prll)
            v_orthog_ego_frame = np.dot(other_velocity, host_ref_orth)
            # Calculate time to impact if required
            time_to_impact = float('inf')
            if sorting_criteria == "time of impact":
                combined_radius = host_radius + other_radius
                time_impact = compute_time_to_impact(
                    host_position, other_position, host_velocity, other_velocity, combined_radius
                )
                if time_impact is not None:
                    time_to_impact = time_impact
            # Append to the list
            uav_state = [
                p_parallel_ego_frame,
                p_orthog_ego_frame,
                v_parallel_ego_frame,
                v_orthog_ego_frame,
                other_radius,
                host_radius + other_radius,
                dist_to_other
            ]
            uav_state_list.append((uav_state, time_to_impact))
        # Sort based on criteria
        if sorting_criteria == 'closest first':
            uav_state_list.sort(key=lambda x: x[0][6])  # Sort by distance
        elif sorting_criteria == 'closest last':
            uav_state_list.sort(key=lambda x: x[0][6], reverse=True)  # Reverse sort
        elif sorting_criteria == 'time of impact':
            uav_state_list.sort(key=lambda x: x[1])  # Sort by time to impact
        # Extract the sorted states
        sorted_states = [state[0] for state in uav_state_list]
        num_other_agents = len(sorted_states)
        # Fill the pre-allocated array with sorted states
        for i, state in enumerate(sorted_states):
            if i < max_number_other_agents_observed:
                other_uav_states[i] = state
    # Create the transformed data dictionary
    transformed_data = {
        'dist_goal': host_data['distance_to_goal'],
        'heading_ego_frame': host_heading,
        'current_speed': host_data['current_speed'],
        'radius': host_radius,
        'no_other_agents': num_other_agents,  # Number of other agents observed
        'other_agent_state': other_uav_states
    }
    return transformed_data


def transform_for_graph(data, max_number_other_agents_observed) -> Dict:
    # Host UAV (AutoUAV)
    host_agent = data[0]
    host_pos_x = host_agent['current_position'].x
    host_pos_y = host_agent['current_position'].y
    host_vel_x = host_agent['current_speed'] * np.cos(host_agent['current_heading'])
    host_vel_y = host_agent['current_speed'] * np.sin(host_agent['current_heading'])
    host_radius = host_agent['radius']
    host_node_features = [host_pos_x, host_pos_y, host_vel_x, host_vel_y, host_radius]
    # Initialize arrays for graph components
    node_features = np.zeros((max_number_other_agents_observed + 1, 5), dtype=np.float32)
    node_features[0] = host_node_features  # Set host features
    edge_index = np.zeros((2, max_number_other_agents_observed), dtype=np.int64)
    edge_attr = np.zeros((max_number_other_agents_observed, 1), dtype=np.float32)
    mask = np.zeros(max_number_other_agents_observed + 1, dtype=np.float32)
    mask[0] = 1.0  # Host is always valid
    other_uav_data = data[1:] if len(data) > 1 else []
    num_other_agents = len(other_uav_data)
    # Process other UAVs
    for idx, other_uav in enumerate(other_uav_data):
        if idx >= max_number_other_agents_observed:
            break  # Don't exceed maximum number of other agents
        other_pos_x = other_uav['other_uav_current_position'].x
        other_pos_y = other_uav['other_uav_current_position'].y
        other_vel_x = other_uav['other_uav_current_speed'] * np.cos(other_uav['other_uav_current_heading'])
        other_vel_y = other_uav['other_uav_current_speed'] * np.sin(other_uav['other_uav_current_heading'])
        other_radius = other_uav['other_uav_radius']
        # Node features for other UAVs
        node_features[idx + 1] = [other_pos_x, other_pos_y, other_vel_x, other_vel_y, other_radius]
        mask[idx + 1] = 1.0  # Valid node
        # Edge features: Connect host UAV to other UAVs
        distance = host_agent['current_position'].distance(other_uav['other_uav_current_position'])
        edge_index[:, idx] = [0, idx + 1]  # Edge from host to other UAV
        edge_attr[idx, 0] = distance
    # Create the transformed data dictionary
    transformed_data = {
        'num_other_agents': np.array([num_other_agents], dtype=np.int64),
        'agent_dist_to_goal': np.array([host_agent['distance_to_goal']], dtype=np.float32),
        'agent_end_point': np.array([host_agent['end'].x, host_agent['end'].y], dtype=np.float32),
        'agent_current_position': np.array([host_pos_x, host_pos_y], dtype=np.float32),
        'graph_feat_matrix': node_features,  # [num_nodes, feature_dim]
        'edge_index': edge_index,            # [2, num_edges]
        'edge_attr': edge_attr,              # [num_edges, edge_feature_dim]
        'mask': mask,                        # [num_nodes]
    }
    return transformed_data


def transform_for_uam(data) -> Dict:
    # Create the transformed data dictionary
    transformed_data = {}
    # Auto UAV aka Host data
    host_data = data[0]
    other_uav_data = data[1:] if len(data) > 1 else []
    # Process closest intruder if any exist
    if other_uav_data:
        # Sort by distance
        sorted_uavs = sorted(
            other_uav_data,
            key=lambda x: host_data['current_position'].distance(x['other_uav_current_position'])
        )
        # Get closest intruder data
        closest_intruder = sorted_uavs[0]
        # Set intruder detection flag
        intruder_detected = 1
        # Calculate distance to intruder
        distance_to_intruder = host_data['current_position'].distance(closest_intruder['other_uav_current_position'])
        # Calculate relative heading (angle between agent's heading and vector to intruder)
        intruder_vector = np.array([
            closest_intruder['other_uav_current_position'].x - host_data['current_position'].x,
            closest_intruder['other_uav_current_position'].y - host_data['current_position'].y
        ])
        intruder_angle = math.atan2(intruder_vector[1], intruder_vector[0])
        relative_heading_intruder = ((intruder_angle - host_data['current_heading'] + 180) % 360) - 180
        # Intruder's current heading
        intruder_current_heading = closest_intruder['other_uav_current_heading']
    # Get restricted airspace data
    restricted_airspace_detected = host_data.get('static_collision_detected')
    distance_to_restricted_airspace = host_data.get('distance_to_restricted')
    #!TODO fix this to calculuate correctly
    # Calculate relative heading to restricted airspace (default to 0 if no detection)
    relative_heading_restricted_airspace = 0.0
    """Corrected method will be as follows"""
    # if detection == 0:
        # distance_to_ra = 0.0
        # relative_heading_ra = 0.0
    # else:
        # distance_to_ra = host_data.get('distance_to_restricted)
        # relative_heading_ra = Shapely method to find relative heading
    transformed_data = {
        'agent_id': host_data['agent_id'], #! need to check data to see if this is present 
        'agent_dist_to_goal': host_data['distance_to_goal'],
        'agent_speed': host_data['current_speed'],
        'agent_current_heading': host_data['current_heading'],
        'agent_deviation': host_data['deviation'], #! need to check data to see if this is present 
        'intruder_detected': intruder_detected,
        'distance_to_intruder': distance_to_intruder,
        'relative_heading_intruder': relative_heading_intruder,
        'intruder_current_heading': intruder_current_heading,
        'restricted_airspace_detected': restricted_airspace_detected,
        'distance_to_restricted_airspace': distance_to_restricted_airspace,
        'relative_heading_restricted_airspace': relative_heading_restricted_airspace
    }
    return transformed_data