from typing import List, Dict
import numpy as np
from utils import compute_time_to_impact


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
    if return_data_format == 'seq':
        transformed_data = transform_for_sequence(data, sorting_criteria, max_number_other_agents_observed)
        return transformed_data
    elif return_data_format == 'graph':
        transformed_data = transform_for_graph(data, max_number_other_agents_observed)
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