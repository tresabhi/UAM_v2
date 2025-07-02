from typing import List, Dict, Tuple
import numpy as np
from utils import compute_time_to_impact
import math
from gymnasium.spaces import Dict, Box, Discrete, MultiBinary

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
    elif obs_space_string == 'UAV_5_intruders':
        return obs_space_uav_5_intruders(max_number_intruders=5)
    else:
        raise RuntimeError('Incorrect obs_space_string passed to choose_obs_space_constructor')

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

    return Dict(
            {
                # agent speed
                # normalize between 0 and 1 : max-min normalization
                "agent_speed": Box(
                    low=0,
                    high=1,
                    shape=(1,),
                    dtype=np.float32,  # Changed to float32 for consistency
                ),
                # agent current heading
                # normalize between -1 and 1 : max-min normalization
                "agent_current_heading": Box(
                    low=-1,
                    high=1,
                    shape=(1,),  # Changed to (1,) for consistency
                    dtype=np.float32,
                ),
                # agent deviation
                # normalize between -1 and 1 : max-min normalization
                "agent_deviation": Box(
                    low=-1,
                    high=1,
                    shape=(1,),
                    dtype=np.float32,  # Changed to float32 for consistency
                ),
                # agent distance to goal
                # normalize between 0 and 1 : max-min normalization
                "agent_dist_to_goal": Box(
                    low=0,
                    high=1,
                    shape=(1,),
                    dtype=np.float32  # Changed to float32 for consistency
                ),
                # intruder detection
                "intruder_detected": Discrete(
                    2  # 0 for no intruder, 1 for intruder detected
                ),

                # intruder current position x and y
                # normalize between -1 and 1 : max-min normalization
                # 'intruder_position_x': Box(  # Split into x and y components
                #     low=-10000000,
                #     high=10000000,
                #     shape=(1,),
                #     dtype=np.float32,
                # ),
                # 'intruder_position_y': Box(  # Split into x and y components
                #     # normalize between -1 and 1 : max-min normalization
                #     low=-10000000,
                #     high=10000000,
                #     shape=(1,),
                #     dtype=np.float32,
                # ),
                # distance to intruder
                # normalize between 0 and 1 : max-min normalization
                #TODO: this should be limited to the detection radius of 500m
                "distance_to_intruder": Box(
                    low=0,
                    high=1000,  # Increased to match other scales
                    shape=(1,),
                    dtype=np.float32,  # Changed to float32 for consistency
                ),
                # Relative heading of intruder
                # normalize between -1 and 1 : max-min normalization
                "relative_heading_intruder": Box(
                    low=-1, 
                    high=1, 
                    shape=(1,), 
                    dtype=np.float32  # Changed to float32 for consistency
                ),
                # intruder's current heading
                # normalize between -1 and 1 : max-min normalization
                # "intruder_current_heading": Box(
                #     low=-180, 
                #     high=180, 
                #     shape=(1,), 
                #     dtype=np.float32,  # Changed to float32 for consistency
                # ),
                # relative speed
                # normalize between -1 and 1 : max-min normalization
                "relative_intruder_speed": Box(
                    low=-1,
                    high=1,  # Reduced to realistic value
                    shape=(1,),
                    dtype=np.float32,  # Changed to float32 for consistency
                ),
                # intruder speed
                # normalize between 0 and 1 : max-min normalization
                # "intruder_speed": Box(
                #     low=0,
                #     high=100,  # Reduced to realistic value
                #     shape=(1,),
                #     dtype=np.float32,  # Changed to float32 for consistency
                # ),                                               
                # restricted airspace
                "restricted_airspace_detected":Discrete(
                    2 # 0 for no ra, 1 for ra detected
                ),
                # distance to airspace 
                # normalize between 0 and 1 : max-min normalization
                "distance_to_restricted_airspace": Box(
                    low=0,
                    high=1,
                    shape=(1,),
                    dtype=np.float32,  # Changed to float32 for consistency
                ),
                # relative heading restricted airspace
                # normalize between -1 and 1 : max-min normalization
                "relative_heading_restricted_airspace": Box(
                    low=-1,
                    high=1,
                    shape=(1,),
                    dtype=np.float32,  # Changed to float32 for consistency
                )
            }
        )



def obs_space_uav_5_intruders(max_number_intruders=5):
    '''Obs space for up to 5 intruders'''
    # TODO: need to add a few more fields
    # 1. host/auto_UAV bearing to destination
    # 2. might need to add bearing for other UAVs too  
    return Dict({
        'dist_goal': Box(low=0, high=10000, shape=(), dtype=np.float32),
        'heading_ego_frame': Box(low=-180, high=180, shape=(), dtype=np.float32),
        'current_speed': Box(low=0, high=50, shape=(), dtype=np.float32),
        'radius': Box(low=0, high=20, shape=(), dtype=np.float32),  # UAV size
        'no_other_agents_detected': Discrete(max_number_intruders + 1),  # Number of
        'other_uav_mask': MultiBinary(max_number_intruders),
        'other_agents_state': Box(  # p_parall, p_orth, v_parall, v_orth, other_agent_radius, combined_radius, dist_2_other
            low=np.full((max_number_intruders, 7), -np.inf),
            high=np.full((max_number_intruders, 7), np.inf),
            shape=(max_number_intruders, 7),
            dtype=np.float32,
        ),        
        # Static object detection
        'restricted_area_mask':MultiBinary(1),
        'static_collision_detected': Box(low=0, high=1, shape=(), dtype=np.int32),
        'distance_to_restricted': Box(low=0, high=np.inf, shape=(), dtype=np.float32),
    })

#### DATA TRANSFORMATION ####
def transform_sensor_data(data,
                          max_number_other_agents_observed,
                          return_data_format:str, #TODO: needs a better name that reflects its purpose
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
    elif return_data_format == 'UAV_5_intruders':
        transformed_data = transform_for_uav_5_intruders(data, sorting_criteria, max_number_other_agents_observed)
        return transformed_data
    else:
        raise RuntimeError('Incorrect data_format passed to transform_sensor_data')


# TODO: normalize the return data
def transform_for_uav_5_intruders(data: Tuple[Dict, Tuple[List[Dict], List[Dict]]], 
                                  sorting_criteria, 
                                  max_number_intruders):
    """Transform UAV data into observation for RL training, ensuring all values are consistently shaped"""
    #       host_data,  (other_uavs, restricted_areas)
    # data -> dict, tuple(list,         list); dict is host data, tuple[0]-other_uav, tuple[1]-restricted area
    transformed_data = {}
    
    # Check data format and extract host and other UAV data
    if isinstance(data, tuple):
        # Format: (host_data, (other_agents_list, restricted_areas_list))
        host_data = data[0]
        # other_uav_data -> List[Dict]
        other_uav_data = data[1][0] #if len(data[1][0]) > 0 else [] #TODO: this if else is not needed 
        # You can also extract restricted areas data if needed
        # restricted_areas -> List[Dict]
        restricted_areas_list = data[1][1]
    
    # TODO: check if this else is ever used
    # else is NOT used in current env implementation
    else:
        # Original list format
        host_data = data[0]
        other_uav_data = data[1:] if len(data) > 1 else []
    
    # Host UAV (AutoUAV)
    host_position = np.array([host_data['current_position'].x,
                              host_data['current_position'].y])
    host_velocity = np.array([host_data['current_speed'] * np.cos(host_data['current_heading']),
                            host_data['current_speed'] * np.sin(host_data['current_heading'])])
    host_heading = np.array(host_data['current_heading'])
    host_radius = np.array(host_data['radius'])
    host_ref_prll = np.array(host_data['ref_prll'])
    host_ref_orth = np.array(host_data['ref_orth'])
    host_dist_goal = np.array(host_data['distance_to_goal'])
    host_current_speed = np.array(host_data['current_speed'])

    # Other UAVs
    #TODO: add mask for intruders 

    # Initialize empty array for other UAVs' states
    other_uav_states = np.zeros((max_number_intruders, 7), dtype=np.float32)
    
    #TODO: place the mask for intruders here 
    other_uav_mask_array = np.zeros(max_number_intruders, dtype=np.int8)

    num_other_agents = np.array(0)

    #resticted area data
    static_collsion_detected = np.array(0)
    distance_to_restricted = np.array(float('inf'))
    restriceted_area_mask = np.zeros(1, dtype=np.int8)
    
    
    # Process other UAVs if any exist
    if other_uav_data:
        other_uav_states_list = []

        for other_uav in other_uav_data:
            other_position = np.array([
                other_uav['other_uav_current_position'].x,
                other_uav['other_uav_current_position'].y
            ])
            other_velocity = np.array([
                other_uav['other_uav_current_speed'] * np.cos(other_uav['other_uav_current_heading']),
                other_uav['other_uav_current_speed'] * np.sin(other_uav['other_uav_current_heading'])
            ])
            other_radius = other_uav['other_uav_radius']

            rel_pos = other_position - host_position
            p_parallel_ego_frame = np.dot(rel_pos, host_ref_prll)
            p_orthog_ego_frame = np.dot(rel_pos, host_ref_orth)
            dist_between_centers = np.linalg.norm(rel_pos)
            dist_to_other = dist_between_centers - host_radius - other_radius

            v_parallel_ego_frame = np.dot(other_velocity, host_ref_prll)
            v_orthog_ego_frame = np.dot(other_velocity, host_ref_orth)

            # TODO: is this attr needed for RL?
            time_to_impact = float('inf')
            if sorting_criteria == "time of impact":
                combined_radius = host_radius + other_radius
                time_impact = compute_time_to_impact(
                    host_position, other_position, host_velocity, other_velocity, combined_radius
                )
                if time_impact is not None:
                    time_to_impact = time_impact

            uav_state = [
                p_parallel_ego_frame,
                p_orthog_ego_frame,
                v_parallel_ego_frame,
                v_orthog_ego_frame,
                other_radius,
                host_radius + other_radius,
                dist_to_other
            ]
            other_uav_states_list.append(uav_state) 
            # other_uav_states_list -> list of lists, [[uav1_state], [uav2_state], ...]
            # other_uav_states_list can have variable length, 0,1,2.....
        
        # Sort local other_uav_states_listbased on criteria
        if sorting_criteria == 'closest first':
            other_uav_states_list.sort(key=lambda x: x[6])  # Sort by distance to other UAV
        elif sorting_criteria == 'closest last':
            other_uav_states_list.sort(key=lambda x: x[6], reverse=True)  # Reverse sort    
        else: 
            raise RuntimeError('Incorrect sorting_criteria passed to transform_for_uav_5_intruders')
        
        num_other_agents = len(other_uav_states_list)
        
        for i, state in enumerate(other_uav_states_list):
            # Fill the pre-allocated array with sorted states
            if i < max_number_intruders:
                other_uav_states[i] = state
                other_uav_mask_array[i] = 1
        
        

        # TODO: this section will fail, test it using multiple resticted areas
        # TODO: also test sensor.get_ra_detection()
        if restricted_areas_list:
            for restriceted_area in restricted_areas_list:
                # logic below needs to be fixed to handle multiple restricted areas
                static_collsion_detected = np.array(1)
                distance_to_restricted = np.array(restriceted_area.get('distance', float('inf')))
                restriceted_area_mask[0] = 1


    else:
        pass
        
    transformed_data = {
        'dist_goal': host_dist_goal, #host_data['distance_to_goal'],
        'heading_ego_frame': host_heading,
        'current_speed': host_current_speed, #host_data['current_speed'],
        'radius': host_radius,
        'no_other_agents_detected': num_other_agents,  # Number of other agents observed
        'other_agents_mask': other_uav_mask_array, 
        'other_agents_state': other_uav_states, #! print this to console and make sure this is correct
        'restricted_area_mask': restriceted_area_mask,
        'static_collision_detected': static_collsion_detected,
        'distance_to_restricted': distance_to_restricted
        }

    return transformed_data


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
        restricted_areas_list = data[1][1]
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
        #                             state is a tuple (uav_state, time_to_impact)
        # sorted_states is a list of uav_state
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
    # if isinstance(data, tuple) and restricted_areas_list and len(restricted_areas_list) > 0:
    if restricted_areas_list:
        for restricted_area in restricted_areas_list:
            static_collision_detected = 1
            distance_to_restricted = restricted_area.get('distance', float('inf'))
    
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
        restricted_areas_list = data[1][1]
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


def transform_for_uam(data):
    """Transform UAV data into observation for RL training, ensuring all values are consistently shaped"""
    # Auto UAV aka Host data
    host_data = data[0] 
    host_deviation = host_data['current_heading'] - host_data['final_heading']  
    host_deviation = (host_deviation + math.pi) % (2 * math.pi) - math.pi
    host_final_heading = host_data['final_heading']
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
        distance_to_intruder = np.array([normalize_zero_one(
            host_data['current_position'].distance(closest_intruder['other_uav_current_position']), 0, 500)],
            dtype=np.float32)
        
        # Get intruder position coordinates
        intruder_position_x = np.array([closest_intruder['other_uav_current_position'].x], dtype=np.float32)
        intruder_position_y = np.array([closest_intruder['other_uav_current_position'].y], dtype=np.float32)
        
        # Calculate relative heading (angle between agent's heading and vector to intruder)
        intruder_vector = np.array([
            closest_intruder['other_uav_current_position'].x - host_data['current_position'].x,
            closest_intruder['other_uav_current_position'].y - host_data['current_position'].y
        ])
        intruder_angle = math.atan2(intruder_vector[1], intruder_vector[0])
        
        # normalized realtive heading
        # TODO: change the bounds in obs space 
        temp_rel_heading_intruder = normalize_minus_one_one(((intruder_angle - host_data['current_heading'] + np.pi) % (2 * np.pi) - np.pi), -math.pi, math.pi)
        relative_heading_intruder = np.array([temp_rel_heading_intruder ],dtype=np.float32) 
        
        
        # Intruder's current heading
        intruder_current_heading = np.array([closest_intruder['other_uav_current_heading']], dtype=np.float32)
        intruder_speed = np.array([closest_intruder['other_uav_current_speed']], dtype=np.float32)
        # speed_intruder_relative_to_host = v_int_wrt_earth - v_host_wrt_earth (relative speed in 1D: v_a_earth = v_a_b + v_b_earth)
        intruder_relative_speed = np.array([normalize_minus_one_one(closest_intruder['other_uav_current_speed'] - host_data['current_speed'], 0, host_data['max_speed'])], dtype=np.float32)
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
        ra_distance = np.array([normalize_zero_one(closest_ra['distance'], 0, host_data['detection_radius'])], dtype=np.float32)
        ra_heading = np.array([normalize_minus_one_one(closest_ra['ra_heading'], -math.pi, math.pi)], dtype=np.float32)
    else:
        ra_detected = np.array([0], dtype=np.float32)
        ra_distance = np.array([0.0], dtype=np.float32)
        ra_heading = np.array([0.0], dtype=np.float32)
    
    # Ensure host data is properly formatted as numpy arrays
    agent_id = np.array([host_data['id']], dtype=np.int64)
    agent_speed = np.array([normalize_zero_one(host_data['current_speed'], host_data['min_speed'], host_data['max_speed'])], dtype=np.float32)
    agent_current_heading = np.array([normalize_minus_one_one(host_data['current_heading'], -math.pi, math.pi)], dtype=np.float32)
    agent_deviation = np.array([normalize_minus_one_one(host_deviation, -math.pi, math.pi)], dtype=np.float32)
    agent_dist_to_goal = np.array([normalize_zero_one(host_data['distance_to_goal'], 0, host_data['max_dist'])], dtype=np.float32)
    
    # Create the transformed data dictionary with consistent numpy arrays
    transformed_data = {
        'agent_speed': agent_speed,
        'agent_current_heading': agent_current_heading, 
        
        'agent_deviation': agent_deviation,
        'agent_dist_to_goal': agent_dist_to_goal,
        
        'intruder_detected': intruder_detected,
        'distance_to_intruder': distance_to_intruder,

        'relative_heading_intruder': relative_heading_intruder,

        'relative_intruder_speed': intruder_relative_speed,

        
        'restricted_airspace_detected': ra_detected,
        'distance_to_restricted_airspace': ra_distance,
        'relative_heading_restricted_airspace': ra_heading
    }
    
    return transformed_data



def normalize_zero_one(val, min_val, max_val):
    normal_val = (val - min_val)/(max_val - min_val)
    return normal_val


def normalize_minus_one_one(val, min_val, max_val):
    normal_val = 2 * normalize_zero_one(val, min_val, max_val) - 1
    return normal_val
    

