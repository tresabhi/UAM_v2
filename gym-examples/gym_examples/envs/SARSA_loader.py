import json
import os
from typing import List, Dict, Tuple, Union
import numpy as np

class TrajectoryAnalyzer:
    def __init__(self, log_dir: str = "sarsa_logs"):
        """Initialize the trajectory analyzer
        
        Args:
            log_dir (str): Directory containing the SARSA log files
        """
        self.log_dir = log_dir
        
    def load_specific_file(self, filename: str) -> Dict:
        """Load a specific trajectory file by name
        
        Args:
            filename (str): Name of the file to load (e.g., 'episode_0_20250214_142128.json')
            
        Returns:
            Dict containing the episode data
        """
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def list_available_files(self) -> Dict[str, List[str]]:
        """List all available trajectory files, categorized by outcome
        
        Returns:
            Dict with 'successful' and 'unsuccessful' lists of filenames
        """
        successful = []
        unsuccessful = []
        
        for filename in os.listdir(self.log_dir):
            if filename.endswith('.json'):
                data = self.load_specific_file(filename)
                if data['success']:
                    successful.append(filename)
                else:
                    unsuccessful.append(filename)
                    
        return {
            'successful': successful,
            'unsuccessful': unsuccessful
        }
    
    def get_trajectory_info(self, filename: str) -> Dict:
        """Get basic information about a trajectory
        
        Args:
            filename (str): Name of the file to analyze
            
        Returns:
            Dict containing trajectory information
        """
        data = self.load_specific_file(filename)
        return {
            'episode_id': data['episode_id'],
            'timestamp': data['timestamp'],
            'success': data['success'],
            'truncated': data['truncated'],
            'num_transitions': len(data['transitions']),
            'start_state': data['transitions'][0]['state'],
            'final_state': data['transitions'][-1]['state']
        }
    
    def get_transition_at_step(self, filename: str, step: int) -> Dict:
        """Get a specific transition from a trajectory
        
        Args:
            filename (str): Name of the file to analyze
            step (int): Step number to retrieve
            
        Returns:
            Dict containing the transition data
        """
        data = self.load_specific_file(filename)
        if step >= len(data['transitions']):
            raise IndexError(f"Step {step} out of range. Trajectory has {len(data['transitions'])} steps.")
        return data['transitions'][step]
    
    def get_transition_range(self, filename: str, start_step: int, end_step: int) -> List[Dict]:
        """Get a range of transitions from a trajectory
        
        Args:
            filename (str): Name of the file to analyze
            start_step (int): Starting step number (inclusive)
            end_step (int): Ending step number (exclusive)
            
        Returns:
            List of transition dictionaries
        """
        data = self.load_specific_file(filename)
        return data['transitions'][start_step:end_step]
    
    def get_full_trajectory(self, filename: str) -> Tuple[List, List, List, List, List]:
        """Get a complete trajectory split into components, regardless of success
        
        Args:
            filename (str): Name of the file to analyze
            
        Returns:
            States: List of state dictionaries
            Actions: List of action arrays
            Rewards: List of reward values
            Next_states: List of next state dictionaries
            Next_actions: List of next action arrays
        """

        data = self.load_specific_file(filename)
        transitions = data['transitions']
        
        # Extract components
        states = [t['state'] for t in transitions]
        actions = [t['action'] for t in transitions]
        rewards = [t['reward'] for t in transitions]
        next_states = [t['next_state'] for t in transitions]
        next_actions = [t['next_action'] for t in transitions]
        
        return states, actions, rewards, next_states, next_actions
    
    def print_trajectory_summary(self, filename: str):
        """Print a summary of the trajectory"""
        info = self.get_trajectory_info(filename)
        print(f"\nTrajectory Summary for {filename}:")
        print(f"Episode ID: {info['episode_id']}")
        print(f"Timestamp: {info['timestamp']}")
        print(f"Success: {info['success']}")
        print(f"Truncated: {info['truncated']}")
        print(f"Number of transitions: {info['num_transitions']}")
        print("\nInitial State:")
        self.print_state_details(info['start_state'])
        print("\nFinal State:")
        self.print_state_details(info['final_state'])
    
    def print_state_details(self, state: Dict):
        """Print detailed information about a state"""
        print(f"  Distance to goal: {state['dist_to_goal']:.2f}")
        print(f"  Heading: {state['heading_ego_frame']:.2f}")
        print(f"  Speed: {state['current_speed']:.2f}")
        print(f"  Number of other agents: {state['num_other_agents']}")
        if state['num_other_agents'] > 0:
            print("  Other agents states available in data")
    
    def print_transition_details(self, transition: Dict):
        """Print detailed information about a specific transition"""
        print("\nTransition Details:")
        print(f"State:")
        self.print_state_details(transition['state'])
        
        print(f"\nAction taken: {transition['action']}")
        print(f"Reward: {transition['reward']}")
        
        print(f"\nNext State:")
        self.print_state_details(transition['next_state'])
        
        print(f"\nNext Action: {transition['next_action']}")

if __name__ == "__main__":
    analyzer = TrajectoryAnalyzer()
    
    # List available files
    files = analyzer.list_available_files()
    print("Available trajectory files:")
    print(f"Successful trajectories: {len(files['successful'])}")
    print(f"Unsuccessful trajectories: {len(files['unsuccessful'])}")
    
    # Analyze a specific file (successful or unsuccessful)
    if files['successful'] or files['unsuccessful']:
        # Pick the first available file
        filename = (files['successful'] + files['unsuccessful'])[0]
        
        # Print trajectory summary
        analyzer.print_trajectory_summary(filename)
        
        # Look at specific steps
        print("\nFirst step details:")
        first_transition = analyzer.get_transition_at_step(filename, 0)
        analyzer.print_transition_details(first_transition)
        
        # Get and display a specified range of steps
        start_range = 10, end_range = 15
        print(f"\nGetting steps {start_range}-{end_range}:")
        transitions = analyzer.get_transition_range(filename, 10, 15)
        print(f"Retrieved {len(transitions)} transitions")
        print(f"\nDetailed view of transitions {start_range}-{end_range}:")
        for i, trans in enumerate(transitions):
            print(f"\nStep {i+10}:")
            analyzer.print_transition_details(trans)
        
        # Get and display portions of full trajectory
        states, actions, rewards, next_states, next_actions = analyzer.get_full_trajectory(filename)
        print(f"\nFull trajectory has {len(states)} steps")
        print("\nSample of trajectory data:")
        print("\nFirst 3 states:")
        for i, state in enumerate(states[:3]):
            print(f"\nState at step {i}:")
            analyzer.print_state_details(state)
        
        print("\nFirst 3 actions:")
        for i, action in enumerate(actions[:3]):
            print(f"\nAction at step {i}: {action}")
            
        print("\nRewards (first 3):")
        for i, reward in enumerate(rewards[:3]):
            print(f"Step {i} reward: {reward}")

    ###################################################################
    # Example usage with a speceific file to access trajectory arrays #
    ###################################################################

    # analyzer = TrajectoryAnalyzer()

    # # Get specific range of transitions
    # transitions = analyzer.get_transition_range("episode_0_20250214_142128.json", 10, 15)
    # for trans in transitions:
    #     print(f"Distance to goal: {trans['state']['dist_to_goal']}")
    #     print(f"Action taken: {trans['action']}")

    # # Get full trajectory data
    # states, actions, rewards, next_states, next_actions = analyzer.get_full_trajectory("episode_0_20250214_142128.json")

    # # Access specific timestep
    # timestep = 5
    # print(f"State at timestep {timestep}:")
    # print(f"Distance to goal: {states[timestep]['dist_to_goal']}")
    # print(f"Action taken: {actions[timestep]}")
    # print(f"Reward received: {rewards[timestep]}")