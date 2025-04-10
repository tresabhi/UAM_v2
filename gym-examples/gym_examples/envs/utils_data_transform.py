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
    host_heading = np.array(host_data['current_heading'])
    host_radius = host_data['radius']
    host_ref_prll = np.array([host_data['ref_prll']])
    host_ref_orth = np.array([host_data['ref_orth']])

    if len(data) > 1:
        other_uav_data = data[1:]  # Skip host agent

        # List to store relative states of other UAVs
        other_uav_states = []
        # Other UAV data
        for other_uav in other_uav_data:
            other_position = np.array(other_uav['other_uav_current_position'].x, 
                                    other_uav['other_uav_current_position'].y)
            other_velocity = np.array(other_uav['other_uav_current_speed']*np.cos(other_uav['other_uav_current_heading']), 
                                    other_uav['other_uav_current_speed']*np.sin(other_uav['other_uav_current_heading']))
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
            time_to_impact = None
            if sorting_criteria == "time of impact":
                combined_radius = host_radius + other_radius
                time_to_impact = compute_time_to_impact(
                    host_position, other_position, host_velocity, other_velocity, combined_radius
                )

            # Append to the list
            other_uav_states.append([
                p_parallel_ego_frame,
                p_orthog_ego_frame,
                v_parallel_ego_frame,
                v_orthog_ego_frame,
                other_radius,
                host_radius + other_radius,
                dist_to_other,
                time_to_impact
            ])

        # Sort based on criteria
        if sorting_criteria == 'closest first':
            other_uav_states = sorted(other_uav_states, key=lambda x: x[6])  # Sort by distance
            other_uav_states = [uav_state[:7] for uav_state in other_uav_states]
            
        elif sorting_criteria == 'closest last':
            other_uav_states = sorted(other_uav_states, key=lambda x: x[6], reverse=True)  # Reverse sort
            other_uav_states = [uav_state[:7] for uav_state in other_uav_states]
            
        elif sorting_criteria == 'time of impact':
            other_uav_states = sorted(other_uav_states, key=lambda x: (x[7] or float('inf'), x[6]))
            other_uav_states = [uav_state[:7] for uav_state in other_uav_states]
    else:
        #FIX: make a other_agents_states matrix with zeros, i know there should be 6 features, but how many other_agents ie datapoints should I include 
        other_uav_states = []
    
    
        

    # Generate a mask for valid entries
    num_other_agents = len(other_uav_states)
    mask = [1] * num_other_agents

    # Truncate or pad to ensure fixed size
    if num_other_agents > max_number_other_agents_observed:
        other_uav_states = other_uav_states[:max_number_other_agents_observed]
        mask = mask[:max_number_other_agents_observed]
    elif num_other_agents < max_number_other_agents_observed:
        padding = [[0] * 7] * (max_number_other_agents_observed - num_other_agents) #! NOT 7 - it should be 8, there are 8 features of other_UAV
        other_uav_states.extend(padding)
        mask.extend([0] * (max_number_other_agents_observed - num_other_agents))

    other_uav_states = np.array(other_uav_states)
    mask = np.array(mask, dtype=np.float32)  # Ensure the mask is a NumPy array

    # Create the transformed data dictionary
    transformed_data = {
        'dist_to_goal': host_data['distance_to_goal'],
        'heading_ego_frame': host_data['current_heading'],
        'current_speed': host_data['current_speed'],
        'radius': host_radius,
        'num_other_agents': num_other_agents,  # Number of other agents observed
        'mask': mask,  # Mask for valid entries
        'other_agents_states': other_uav_states
    }
    
    return transformed_data

##### USAGE OF MASK IN PYTORCH #####
# import torch
# import torch.nn.utils.rnn as rnn_utils

# # Example data
# transformed_data = transform_for_sequence(data, max_number_other_agents_observed, sorting_criteria)
# other_agents_states = torch.tensor(transformed_data['other_agents_states'], dtype=torch.float32)
# mask = torch.tensor(transformed_data['mask'], dtype=torch.float32)

# # Use mask to pack sequences for LSTM
# packed_states = rnn_utils.pack_padded_sequence(other_agents_states, mask.sum(dim=-1).int(), batch_first=True, enforce_sorted=False)

# # Forward pass through LSTM
# lstm = torch.nn.LSTM(input_size=7, hidden_size=64, batch_first=True)
# output, _ = lstm(packed_states)

# # Unpack the sequence
# output, _ = rnn_utils.pad_packed_sequence(output, batch_first=True)

##### ---------- END ---------- ##### 

def transform_for_graph(data, max_number_other_agents_observed) -> Dict:
    # Host UAV (AutoUAV)
    host_agent = data[0]
    host_pos_x = host_agent['current_position'].x
    host_pos_y = host_agent['current_position'].y
    host_vel_x = host_agent['current_speed'] * np.cos(host_agent['current_heading'])
    host_vel_y = host_agent['current_speed'] * np.sin(host_agent['current_heading'])
    host_radius = host_agent['radius']

    host_node_features = [host_pos_x, host_pos_y, host_vel_x, host_vel_y, host_radius]

    other_uav_data = data[1:]

    # Initialize lists for graph components
    node_features = [host_node_features]  # Include host UAV features
    edge_index = []  # Edge connections
    edge_attr = []   # Edge attributes (e.g., distances)
    mask = [1]       # Mask for the host UAV (always valid)

    # Process other UAVs
    for idx, other_uav in enumerate(other_uav_data):
        other_pos_x = other_uav['other_uav_current_position'].x
        other_pos_y = other_uav['other_uav_current_position'].y
        other_vel_x = other_uav['other_uav_current_speed'] * np.cos(other_uav['other_uav_current_heading'])
        other_vel_y = other_uav['other_uav_current_speed'] * np.sin(other_uav['other_uav_current_heading'])
        other_radius = other_uav['other_uav_radius']

        # Node features for other UAVs
        other_node_features = [other_pos_x, other_pos_y, other_vel_x, other_vel_y, other_radius]
        node_features.append(other_node_features)
        mask.append(1)  # Valid node

        # Edge features: Connect host UAV to other UAVs
        host_to_other_edge = [
            host_agent['current_position'].distance(other_uav['other_uav_current_position'])  # Distance
        ]
        edge_index.append([0, idx + 1])  # Edge from host to other UAV
        edge_attr.append(host_to_other_edge)

    # Handle padding for missing UAVs
    num_other_agents = len(other_uav_data)
    padding_nodes = max_number_other_agents_observed - num_other_agents
    for _ in range(padding_nodes):
        node_features.append([0] * 5)  # Pad node features with zeros
        mask.append(0)  # Mark as invalid
    for _ in range(padding_nodes):
        edge_attr.append([0])  # Pad edge attributes
    edge_index += [[0, 0]] * padding_nodes  # Pad edge index (self-loops or dummy edges)

    # Convert lists to NumPy arrays
    node_features = np.array(node_features, dtype=np.float32)
    edge_index = np.array(edge_index[:max_number_other_agents_observed], dtype=np.int64).T if edge_index else np.empty((2, 0), dtype=np.int64)
    edge_attr = np.array(edge_attr[:max_number_other_agents_observed], dtype=np.float32)

    # Create the transformed data dictionary
    transformed_data = {
        'num_other_agents': num_other_agents,       # Number of valid other agents
        'agent_dist_to_goal': host_agent['distance_to_goal'],
        'agent_end_point': np.array(host_agent['end'], dtype=np.float32),
        'agent_current_position': np.array(host_agent['current_position'], dtype=np.float32),
        'graph_feat_matrix': node_features,  # [num_nodes, feature_dim]
        'edge_index': edge_index,            # [2, num_edges]
        'edge_attr': edge_attr,              # [num_edges, edge_feature_dim]
        'mask': np.array(mask, dtype=np.float32),  # [num_nodes]
    }
    
    return transformed_data



def process_obs(obs:Dict): 
    '''Process transformed obs and make three arrays.
    1. - learning agent obs array
    2. - other agents obs array
    3. - mask'''
    # obs -> dict('learning_agent_attribute'........,
    #             'other_agents_states':[[1,2,3],[4,5,6], ....])

    learning_agent_states = []

    for obs_keys, obs_value in obs.items():
        if obs_keys != 'other_agents_states' and obs_keys != 'mask':
            learning_agent_states.append(obs_value)
        elif obs_keys == 'other_agents_states' and obs_keys != 'mask':
            other_agents_states = obs_value
        else:
            mask = obs_value
    
    return learning_agent_states, other_agents_states, mask 




