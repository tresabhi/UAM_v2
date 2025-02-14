import json
import os
from datetime import datetime
import numpy as np

class SARSALogger:
    """Logger class for storing SARSA transitions in UAM environment"""
    
    def __init__(self, log_dir="sarsa_logs"):
        """Initialize the SARSA logger
        
        Args:
            log_dir (str): Directory to store the log files
        """
        self.log_dir = log_dir
        self.current_episode = []
        self.episode_count = 0
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize timestamp for the logging session
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _serialize_array(self, arr):
        """Convert numpy arrays to lists for JSON serialization"""
        if isinstance(arr, np.ndarray):
            return arr.tolist()
        elif isinstance(arr, dict):
            return {k: self._serialize_array(v) for k, v in arr.items()}
        elif isinstance(arr, (list, tuple)):
            return [self._serialize_array(item) for item in item]
        return arr
    
    def log_transition(self, state, action, reward, next_state, next_action, info=None):
        """Log a single SARSA transition
        
        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received (can be None)
            next_state: Next state observation
            next_action: Next action to be taken
            info: Additional information dictionary
        """
        transition = {
            'state': self._serialize_array(state),
            'action': self._serialize_array(action),
            'reward': reward,
            'next_state': self._serialize_array(next_state),
            'next_action': self._serialize_array(next_action),
            'info': info if info is not None else {}
        }
        self.current_episode.append(transition)
    
    def end_episode(self, success=False, truncated=False):
        """End the current episode and save it to file
        
        Args:
            success (bool): Whether the episode ended successfully
            truncated (bool): Whether the episode was truncated
        """
        if not self.current_episode:
            return
            
        episode_data = {
            'episode_id': self.episode_count,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'success': success,
            'truncated': truncated,
            'transitions': self.current_episode
        }
        
        # Create filename with episode information
        filename = f"episode_{self.episode_count}_{self.timestamp}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(episode_data, f, indent=2)
            
        # Reset episode data and increment counter
        self.current_episode = []
        self.episode_count += 1
        
    def close(self):
        """Close the logger and save any remaining data"""
        if self.current_episode:
            self.end_episode()