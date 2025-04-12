from typing import List, Dict, Tuple
import numpy as np
from utils import compute_time_to_impact
import math
from gymnasium.spaces import Dict, Box, Discrete

def choose_obs_space_constructor(obs_space_string:str, max_number_other_agents_observed=7):
    '''This helper method uses the argument to return correct obs_space constructor for gym env
        Args:
            obs_space_string
        
        Returns:
            function (object)'''
    if obs_space_string == 'LSTM-A2C':
        return obs_space_seq(max_number_other_agents_observed)
    elif obs_space_string == 'GNN-A2C':
        return obs_space_graph(max_number_other_agents_observed)
    elif obs_space_string == 'UAM_UAV':
        return obs_space_uam()

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
def obs_space_uam():
    '''Obs space for one intruder and restricted area'''
    # return Dict(
    #         {
    #             # agent ID for debugging
    #             "agent_id": Box(
    #                 low=0,
    #                 high=100000,
    #                 shape=(1,),
    #                 dtype=np.int64,
    #             ),
    #             # agent speed
    #             "agent_speed": Box(  #!TODO ensure that attribute is being properly referenced 
    #                 low=0,
    #                 high=100,
    #                 shape=(1,),
    #                 dtype=np.float64,
    #             ),
    #             "agent_current_heading": Box(
    #                 low=-180,
    #                 high=180,
    #                 shape=(),
    #                 dtype=np.float32,
    #                 ),
    #             # agent deviation
    #             "agent_deviation": Box(
    #                 low=-180,
    #                 high=180,
    #                 shape=(1,),
    #                 dtype=np.float64,
    #             ),
    #             # agent distance to goal
    #             "agent_dist_to_goal": Box(
    #                 low=0,
    #                 high=100000,
    #                 shape=(1,),
    #                 dtype=np.float64
    #             ),
    #             # intruder detection
    #             "intruder_detected": Discrete(
    #                 2  # 0 for no intruder, 1 for intruder detected
    #             ),
    #             # intruder id for debugging
    #             "intruder_id": Box(
    #                 low=0,
    #                 high=100000,
    #                 shape=(1,),
    #                 dtype=np.int64,  #! find if it is possible to create ids that take less space
    #             ),
    #             #intruder_current_position
    #             'intruder_current_position': Box(
    #                 low=-10000000,
    #                 high=10000000,
    #                 shape=(1,),
    #                 dtype=np.float64,
    #             ),
    #             # distance to intruder
    #             "distance_to_intruder": Box(
    #                 low=0,
    #                 high=1000,
    #                 shape=(1,),
    #                 dtype=np.float64,
    #             ),
    #             # Relative heading of intruder #!should this be corrected to -180 to 180,
    #             "relative_heading_intruder": Box(
    #                 low=-180, 
    #                 high=180, 
    #                 shape=(1,), 
    #                 dtype=np.float64
    #             ),
    #             # intruder's current heading
    #             "intruder_current_heading": Box(
    #                 low=-180, 
    #                 high=180, 
    #                 shape=(1,), 
    #                 dtype=np.float64,
    #             ),
    #             # relative speed
    #             "relative_intruder_speed": Box(
    #                 low=0,
    #                 high=100000,
    #                 shape=(1,),
    #                 dtype=np.float64),
    #             # intruder speed
    #             "intruder_speed": Box(
    #                 low=0,
    #                 high=10000,
    #                 shape=(1,),
    #                 dtype=np.float64),                                                 
    #             # restricted airspace
    #             "restricted_airspace_detected":Discrete(
    #                 2 # 0 for no ra, 1 for ra detected
    #             ),
    #             # distance to airspace 
    #             "distance_to_restricted_airspace": Box(
    #                 low=0,
    #                 high=10000,
    #                 shape=(1,),
    #                 dtype=np.float64,
    #             ),
    #             "relative_heading_restricted_airspace": Box(
    #                 low=-180,
    #                 high=180,
    #                 shape=(1,),
    #                 dtype=np.float64,
    #             )
    #         }
    #     )
    return Dict(
            {
                # agent ID for debugging
                "agent_id": Box(
                    low=0,
                    high=100000,
                    shape=(1,),
                    dtype=np.int64,
                ),
                # agent speed
                "agent_speed": Box(
                    low=0,
                    high=100,
                    shape=(1,),
                    dtype=np.float32,  # Changed to float32 for consistency
                ),
                # agent current heading
                "agent_current_heading": Box(
                    low=-180,
                    high=180,
                    shape=(1,),  # Changed to (1,) for consistency
                    dtype=np.float32,
                ),
                # agent deviation
                "agent_deviation": Box(
                    low=-180,
                    high=180,
                    shape=(1,),
                    dtype=np.float32,  # Changed to float32 for consistency
                ),
                # agent distance to goal
                "agent_dist_to_goal": Box(
                    low=0,
                    high=100000,
                    shape=(1,),
                    dtype=np.float32  # Changed to float32 for consistency
                ),
                # intruder detection
                "intruder_detected": Discrete(
                    2  # 0 for no intruder, 1 for intruder detected
                ),
                # intruder id for debugging
                "intruder_id": Box(
                    low=-1,  # Changed to allow -1 for no intruder
                    high=100000,
                    shape=(1,),
                    dtype=np.int64,
                ),
                # intruder current position x and y
                'intruder_position_x': Box(  # Split into x and y components
                    low=-10000000,
                    high=10000000,
                    shape=(1,),
                    dtype=np.float32,
                ),
                'intruder_position_y': Box(  # Split into x and y components
                    low=-10000000,
                    high=10000000,
                    shape=(1,),
                    dtype=np.float32,
                ),
                # distance to intruder
                "distance_to_intruder": Box(
                    low=0,
                    high=10000,  # Increased to match other scales
                    shape=(1,),
                    dtype=np.float32,  # Changed to float32 for consistency
                ),
                # Relative heading of intruder
                "relative_heading_intruder": Box(
                    low=-180, 
                    high=180, 
                    shape=(1,), 
                    dtype=np.float32  # Changed to float32 for consistency
                ),
                # intruder's current heading
                "intruder_current_heading": Box(
                    low=-180, 
                    high=180, 
                    shape=(1,), 
                    dtype=np.float32,  # Changed to float32 for consistency
                ),
                # relative speed
                "relative_intruder_speed": Box(
                    low=0,
                    high=100,  # Reduced to realistic value
                    shape=(1,),
                    dtype=np.float32,  # Changed to float32 for consistency
                ),
                # intruder speed
                "intruder_speed": Box(
                    low=0,
                    high=100,  # Reduced to realistic value
                    shape=(1,),
                    dtype=np.float32,  # Changed to float32 for consistency
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
                    dtype=np.float32,  # Changed to float32 for consistency
                ),
                # relative heading restricted airspace
                "relative_heading_restricted_airspace": Box(
                    low=-180,
                    high=180,
                    shape=(1,),
                    dtype=np.float32,  # Changed to float32 for consistency
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
    
def transform_for_sequence(data, sorting_criteria, max_number_other_agents_observed):
    """
    Transform UAV data into a fixed-size array with relative states for each UAV.
    
    Args:
        data: Either a tuple (host_data, (other_agents_list, restricted_areas_list)) or List[Dict]
        sorting_criteria: Criteria for sorting other agents
        max_number_other_agents_observed: Maximum number of other agents to include
        
    Returns:
        Dict: Transformed UAV data
    """
    transformed_data = {}
    
    # Check data format and extract host and other UAV data
    if isinstance(data, tuple):
        # Format: (host_data, (other_agents_list, restricted_areas_list))
        host_data = data[0]
        other_uav_data = data[1][0] if len(data[1][0]) > 0 else []
        # You can also extract restricted areas data if needed
        restricted_areas_data = data[1][1]
    else:
        # Original list format
        host_data = data[0]
        other_uav_data = data[1:] if len(data) > 1 else []
    
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
    
    # Process other UAVs if any exist
    if other_uav_data and len(other_uav_data) > 0:
        uav_state_list = []
        
        # Process other UAVs
        for other_uav in other_uav_data:
            # Create position as a numpy array with both x and y components
            other_position = np.array([
                other_uav['other_uav_current_position'].x,
                other_uav['other_uav_current_position'].y
            ])
            # Create velocity as a numpy array with both x and y components
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
    
    # Check if static_collision_detected and distance_to_restricted fields should be included
    static_collision_detected = 0
    distance_to_restricted = float('inf')
    
    # If restricted areas data exists and contains something
    if isinstance(data, tuple) and data[1][1] and len(data[1][1]) > 0:
        ra_data = data[1][1]
        static_collision_detected = 1
        distance_to_restricted = ra_data.get('distance', float('inf'))
    
    # Create the transformed data dictionary
    transformed_data = {
        'dist_goal': host_data['distance_to_goal'],
        'heading_ego_frame': host_heading,
        'current_speed': host_data['current_speed'],
        'radius': host_radius,
        'no_other_agents': num_other_agents,  # Number of other agents observed
        'static_collision_detected': static_collision_detected,
        'distance_to_restricted': distance_to_restricted,
        'other_agent_state': other_uav_states
    }
    
    return transformed_data

def transform_for_graph(data, max_number_other_agents_observed) -> Dict:
    # Check data format and extract host and other UAV data
    if isinstance(data, tuple):
        # Format: (host_data, (other_agents_list, restricted_areas_list))
        host_agent = data[0]
        other_uav_data = data[1][0] if len(data[1][0]) > 0 else []
        # You can also extract restricted areas data if needed
        restricted_areas_data = data[1][1]
    else:
        # Original list format
        host_agent = data[0]
        other_uav_data = data[1:] if len(data) > 1 else []
    
    # Host UAV (AutoUAV)
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
    
    # Check if static_collision_detected and distance_to_restricted fields should be included
    static_collision_detected = 0
    distance_to_restricted = float('inf')
    
    # If restricted areas data exists and contains something
    if isinstance(data, tuple) and data[1][1] and len(data[1][1]) > 0:
        ra_data = data[1][1]
        static_collision_detected = 1
        distance_to_restricted = ra_data.get('distance', float('inf'))
    
    # Create the transformed data dictionary
    transformed_data = {
        'num_other_agents': np.array([num_other_agents], dtype=np.int64),
        'agent_dist_to_goal': np.array([host_agent['distance_to_goal']], dtype=np.float32),
        'agent_end_point': np.array([host_agent['end'].x, host_agent['end'].y], dtype=np.float32),
        'agent_current_position': np.array([host_pos_x, host_pos_y], dtype=np.float32),
        'static_collision_detected': np.array([static_collision_detected], dtype=np.int32),
        'distance_to_restricted': np.array([distance_to_restricted], dtype=np.float32),
        'graph_feat_matrix': node_features,  # [num_nodes, feature_dim]
        'edge_index': edge_index,            # [2, num_edges]
        'edge_attr': edge_attr,              # [num_edges, edge_feature_dim]
        'mask': mask,                        # [num_nodes]
    }
    
    return transformed_data

#                               own_data,(    other_agents, restricted_area)
def transform_for_uam(data):
    """Transform UAV data into observation for RL training, ensuring all values are consistently shaped"""
    # Auto UAV aka Host data
    host_data = data[0]
    host_deviation = host_data['current_heading'] - host_data['final_heading']  
    
    # Other agents data
    other_uav_data = data[1][0] 
    
    # Restricted Area data
    ra_data = data[1][1]
    
    # Process closest intruder if any exist
    if len(other_uav_data):
        # Sort by distance
        sorted_uavs = sorted(
            other_uav_data,
            key=lambda x: host_data['current_position'].distance(x['other_uav_current_position'])
        )
        # Get closest intruder data
        closest_intruder = sorted_uavs[0]
        
        # Set intruder detection flag
        intruder_detected = np.array([1], dtype=np.float32)
        intruder_id = np.array([closest_intruder['other_uav_id']], dtype=np.int64)
        
        # Calculate distance to intruder
        distance_to_intruder = np.array(
            [host_data['current_position'].distance(closest_intruder['other_uav_current_position'])], 
            dtype=np.float32
        )
        
        # Get intruder position coordinates
        intruder_position_x = np.array([closest_intruder['other_uav_current_position'].x], dtype=np.float32)
        intruder_position_y = np.array([closest_intruder['other_uav_current_position'].y], dtype=np.float32)
        
        # Calculate relative heading (angle between agent's heading and vector to intruder)
        intruder_vector = np.array([
            closest_intruder['other_uav_current_position'].x - host_data['current_position'].x,
            closest_intruder['other_uav_current_position'].y - host_data['current_position'].y
        ])
        intruder_angle = math.atan2(intruder_vector[1], intruder_vector[0])
        
        relative_heading_intruder = np.array(
            [((intruder_angle - host_data['current_heading'] + np.pi) % (2 * np.pi) - np.pi) * 180 / np.pi], 
            dtype=np.float32
        )
        
        # Intruder's current heading
        intruder_current_heading = np.array([closest_intruder['other_uav_current_heading']], dtype=np.float32)
        intruder_speed = np.array([closest_intruder['other_uav_current_speed']], dtype=np.float32)
        intruder_relative_speed = np.array([abs(host_data['current_speed'] - closest_intruder['other_uav_current_speed'])], dtype=np.float32)
    else:
        intruder_detected = np.array([0], dtype=np.float32)
        intruder_id = np.array([-1], dtype=np.int64)
        distance_to_intruder = np.array([0.0], dtype=np.float32)
        relative_heading_intruder = np.array([0.0], dtype=np.float32)
        intruder_current_heading = np.array([0.0], dtype=np.float32)
        intruder_relative_speed = np.array([0.0], dtype=np.float32)
        intruder_speed = np.array([0.0], dtype=np.float32)
        intruder_position_x = np.array([0.0], dtype=np.float32)
        intruder_position_y = np.array([0.0], dtype=np.float32)
    
    # Get restricted airspace data
    if len(ra_data) > 0:
        # Sort by distance
        closest_ra = sorted(ra_data, key=lambda x: x['distance'])[0] if len(ra_data) > 1 else ra_data[0]
        ra_detected = np.array([1], dtype=np.float32)
        ra_distance = np.array([closest_ra['distance']], dtype=np.float32)
        ra_heading = np.array([closest_ra['ra_heading']], dtype=np.float32)
    else:
        ra_detected = np.array([0], dtype=np.float32)
        ra_distance = np.array([0.0], dtype=np.float32)
        ra_heading = np.array([0.0], dtype=np.float32)
    
    # Ensure host data is properly formatted as numpy arrays
    agent_id = np.array([host_data['id']], dtype=np.int64)
    agent_speed = np.array([host_data['current_speed']], dtype=np.float32)
    agent_current_heading = np.array([host_data['current_heading']], dtype=np.float32)
    agent_deviation = np.array([host_deviation], dtype=np.float32)
    agent_dist_to_goal = np.array([host_data['distance_to_goal']], dtype=np.float32)
    
    # Create the transformed data dictionary with consistent numpy arrays
    transformed_data = {
        'agent_id': agent_id,
        'agent_speed': agent_speed,
        'agent_current_heading': agent_current_heading,
        'agent_deviation': agent_deviation,
        'agent_dist_to_goal': agent_dist_to_goal,
        
        'intruder_detected': intruder_detected,
        'intruder_id': intruder_id,
        'distance_to_intruder': distance_to_intruder,
        'intruder_position_x': intruder_position_x,
        'intruder_position_y': intruder_position_y,
        'relative_heading_intruder': relative_heading_intruder,
        'intruder_current_heading': intruder_current_heading,
        'relative_intruder_speed': intruder_relative_speed,
        'intruder_speed': intruder_speed,
        
        'restricted_airspace_detected': ra_detected,
        'distance_to_restricted_airspace': ra_distance,
        'relative_heading_restricted_airspace': ra_heading
    }
    
    return transformed_data