#mapped_env_util.py
from gymnasium.spaces import Dict, Box, Discrete
import numpy as np

# Define sequential observation space for LSTM
def obs_space_seq(max_number_other_agents_observed):
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


