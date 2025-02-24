import json
import os
from typing import List, Dict, Tuple, Union
import numpy as np

class NonLearningLoader:
    """Loader class for analyzing non-learning agent trajectories"""
    
    def __init__(self, log_dir="non_learning_logs"):
        self.log_dir = log_dir
        
    def list_episodes(self) -> List[str]:
        """List all available episode directories"""
        return [d for d in os.listdir(self.log_dir) 
                if os.path.isdir(os.path.join(self.log_dir, d))]
    
    def get_episode_agents(self, episode_dir: str) -> List[str]:
        """Get list of agent IDs in an episode"""
        episode_path = os.path.join(self.log_dir, episode_dir)
        return [f.split('_')[1] for f in os.listdir(episode_path) 
                if f.startswith('agent_') and f.endswith('_trajectory.json')]
    
    def load_episode_metadata(self, episode_dir: str) -> Dict:
        """Load metadata for a specific episode"""
        metadata_path = os.path.join(self.log_dir, episode_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def load_agent_trajectory(self, episode_dir: str, agent_id: str) -> Dict:
        """Load trajectory for a specific agent in an episode"""
        trajectory_path = os.path.join(self.log_dir, episode_dir, f"agent_{agent_id}_trajectory.json")
        with open(trajectory_path, 'r') as f:
            return json.load(f)
    
    def get_step(self, episode_dir: str, agent_id: str, step: int) -> Dict:
        """Get a specific step from an agent's trajectory
        
        Args:
            episode_dir (str): Name of the episode directory
            agent_id (str): ID of the agent
            step (int): Step number to retrieve
            
        Returns:
            Dict containing state and action for that step
        """
        trajectory = self.load_agent_trajectory(episode_dir, agent_id)
        if step >= len(trajectory['trajectory']):
            raise IndexError(f"Step {step} out of range. Trajectory has {len(trajectory['trajectory'])} steps.")
        return trajectory['trajectory'][step]
    
    def get_step_range(self, episode_dir: str, agent_id: str, start_step: int, end_step: int) -> List[Dict]:
        """Get a range of steps from an agent's trajectory
        
        Args:
            episode_dir (str): Name of the episode directory
            agent_id (str): ID of the agent
            start_step (int): Starting step (inclusive)
            end_step (int): Ending step (exclusive)
            
        Returns:
            List of state-action pairs for the specified range
        """
        trajectory = self.load_agent_trajectory(episode_dir, agent_id)
        return trajectory['trajectory'][start_step:end_step]
    
    def get_full_trajectory(self, episode_dir: str, agent_id: str) -> Tuple[List, List]:
        """Get the complete trajectory for an agent
        
        Returns:
            Tuple of (states, actions)
        """
        trajectory = self.load_agent_trajectory(episode_dir, agent_id)
        states = [step['state'][0] for step in trajectory['trajectory']]
        actions = [step['action'] for step in trajectory['trajectory']]
        return states, actions
    
    def print_step_details(self, step_data: Dict):
        """Print detailed information about a specific step"""
        print("\nState:")
        state = step_data['state'][0]  # Access first (and only) state in the list
        print(f"  Position: {state['current_position']}")
        print(f"  Speed: {state['current_speed']}")
        print(f"  Heading: {state['current_heading']}")
        print(f"  Distance to goal: {state['distance_to_goal']}")
        print(f"\nAction: {step_data['action']}")
    
    def print_trajectory_summary(self, episode_dir: str, agent_id: str = None):
        """Print a summary of trajectories in an episode
        
        Args:
            episode_dir (str): Name of the episode directory
            agent_id (str, optional): If provided, only show summary for this agent
        """
        metadata = self.load_episode_metadata(episode_dir)
        print(f"\nEpisode Summary ({episode_dir}):")
        print(f"Timestamp: {metadata['timestamp']}")
        print(f"Number of agents: {metadata['num_agents']}")
        
        if metadata['collision_occurred']:
            print(f"Collision occurred between agents: {metadata['collision_agents']}")
            print(f"Collision type: {metadata['collision_type']}")
        
        print(f"Completed agents: {metadata['completed_agents']}")
        
        if agent_id:
            # Print details for specific agent
            trajectory = self.load_agent_trajectory(episode_dir, agent_id)
            print(f"\nAgent {agent_id}:")
            print(f"Trajectory length: {trajectory['trajectory_length']}")
            print(f"Completed successfully: {trajectory['completed_successfully']}")
            print(f"Terminated by collision: {trajectory['terminated_by_collision']}")
            
            # Show first and last state-action pairs
            if trajectory['trajectory']:
                print("\nFirst step:")
                self.print_step_details(trajectory['trajectory'][0])
                print("\nLast step:")
                self.print_step_details(trajectory['trajectory'][-1])
        else:
            # Print summary for all agents
            for aid in self.get_episode_agents(episode_dir):
                trajectory = self.load_agent_trajectory(episode_dir, aid)
                print(f"\nAgent {aid}:")
                print(f"Trajectory length: {trajectory['trajectory_length']}")
                print(f"Completed successfully: {trajectory['completed_successfully']}")
                print(f"Terminated by collision: {trajectory['terminated_by_collision']}")

if __name__ == "__main__":
    loader = NonLearningLoader()
    
    # List all available episodes
    episodes = loader.list_episodes()
    print(f"Found {len(episodes)} episodes")
    
    if episodes:
        episode_dir = episodes[0]
        print("\nAnalyzing first episode...")
        
        # Get list of agents
        agents = loader.get_episode_agents(episode_dir)
        print(f"Found {len(agents)} agents in episode")
        
        # Print episode summary
        loader.print_trajectory_summary(episode_dir)
        
        if agents:
            # Analyze first agent
            agent_id = agents[0]
            print(f"\nDetailed analysis of agent {agent_id}:")
            
            # Get specific step
            step_data = loader.get_step(episode_dir, agent_id, 0)
            print("\nFirst step details:")
            loader.print_step_details(step_data)
            
            # Get range of steps
            steps = loader.get_step_range(episode_dir, agent_id, 5, 10)
            print(f"\nSteps 5-10 ({len(steps)} steps):")
            for i, step in enumerate(steps):
                print(f"\nStep {i+5}:")
                loader.print_step_details(step)
            
            # Get full trajectory
            states, actions = loader.get_full_trajectory(episode_dir, agent_id)
            print(f"\nFull trajectory has {len(states)} steps")
            print("\nSample of trajectory data:")
            print("\nFirst 3 steps:")
            for i, state in enumerate(states[:3]):
                print(f"\nState at step {i}:")
                loader.print_step_details(step)
            
            print("\nFirst 3 actions:")
            for i, action in enumerate(actions[:3]):
                print(f"\nAction at step {i}: {action}")

            # Access specific timestep
            timestep = 5
            print(f"\nStates at timestep {timestep}")
            print(f"Distance to goal: {states[timestep]['distance_to_goal']}")
            print(f"Action taken: {actions[timestep]}")