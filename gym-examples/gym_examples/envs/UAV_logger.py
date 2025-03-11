from datetime import datetime
import json
import numpy as np
import os

class NonLearningLogger:
    """Logger class for storing state-action trajectories of non-learning agents"""
    
    def __init__(self, log_dir="non_learning_logs"):
        self.log_dir = log_dir
        self.agent_trajectories = {}  # Dict to store current episode data for each agent
        self.episode_count = 0
        self.completed_agents = set()  # Track which agents have completed their mission
        self.collision_data = None  # Store collision information if it occurs
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize timestamp for the logging session
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _serialize_array(self, arr):
        """Convert numpy arrays and other types to JSON-serializable format"""
        if isinstance(arr, np.ndarray):
            return arr.tolist()
        elif isinstance(arr, dict):
            return {k: self._serialize_array(v) for k, v in arr.items()}
        elif isinstance(arr, (list, tuple)):
            return [self._serialize_array(x) for x in arr]
        elif isinstance(arr, (bool, int, float, str, type(None))):
            return arr
        else:
            return str(arr)
    
    def log_step(self, agent_id, state, action):
        """Log a single step for a non-learning agent"""
        if agent_id not in self.agent_trajectories:
            self.agent_trajectories[agent_id] = []
            
        step_data = {
            'state': self._serialize_array(state),
            'action': self._serialize_array(action)
        }
        self.agent_trajectories[agent_id].append(step_data)

    def mark_agent_complete(self, agent_id):
        """Mark an agent as having completed its mission"""
        self.completed_agents.add(agent_id)
    
    def record_collision(self, collision_ids, collision_type):
        """Record collision information"""
        self.collision_data = {
            'collision_ids': collision_ids,
            'collision_type': collision_type
        }
    
    def save_episode(self):
        """Save all agent trajectories to a single episode directory"""
        if not self.agent_trajectories:
            return
            
        # Create episode directory with timestamp
        episode_dir = os.path.join(self.log_dir, f"episode_{self.timestamp}")
        os.makedirs(episode_dir, exist_ok=True)
        
        # Save metadata for the episode
        metadata = {
            'timestamp': self.timestamp,
            'collision_occurred': self.collision_data is not None,
            'collision_agents': self._serialize_array(self.collision_data['collision_ids'] if self.collision_data else None),
            'collision_type': self.collision_data['collision_type'] if self.collision_data else None,
            'num_agents': len(self.agent_trajectories),
            'completed_agents': list(self.completed_agents)
        }
        
        with open(os.path.join(episode_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save each agent's trajectory to a separate file
        for agent_id, trajectory in self.agent_trajectories.items():
            agent_data = {
                'agent_id': agent_id,
                'trajectory_length': len(trajectory),
                'terminated_by_collision': (self.collision_data and 
                                         agent_id in (self.collision_data['collision_ids'] or [])),
                'completed_successfully': agent_id in self.completed_agents,
                'trajectory': trajectory
            }
            
            filename = f"agent_{agent_id}_trajectory.json"
            filepath = os.path.join(episode_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(agent_data, f, indent=2)
    
    def reset(self):
        """Reset the logger for a new episode"""
        self.agent_trajectories = {}
        self.completed_agents = set()
        self.collision_data = None
        
    def close(self):
        """Save the episode data and close the logger"""
        self.save_episode()