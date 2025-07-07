from typing import List, Tuple, Dict, Optional
import numpy as np

from map_env_revised import MapEnv
from MCTS import *


env = MapEnv(number_of_uav=0,
             num_ORCA_uav=0,
             number_of_vertiport=8,
             location_name='Austin, Texas, USA',
             airspace_tag_list=[],
             vertiport_tag_list=[], 
             max_episode_steps=6000,
             number_of_other_agents_for_model=7,#what is this???
             sleep_time=0,
             seed=70,
             obs_space_str='UAM_UAV',
             sorting_criteria='closest_first',
             render_mode=None,
             max_uavs=100,
             max_vertiports=150,
             make_uav_at_timestep=300)



class VertiportDesignEnv():
    def __init__(self,):
        self.selected_vertiport_list:List = []
        self.start_state:Tuple = self.selected_vertiport_list, len(self.selected_vertiport_list)
        self.end_state:Tuple = self.selected_vertiport_list, len(env.num_regions) #TODO: env needs a num_region init attr
        self.internal_state:Tuple = self.selected_vertiport_list, len(self.selected_vertiport_list)
        self.action_dim = 2 # action = [layer_no<action_idx 0>, vertiports_in_layer<action_idx 1>]

        pass 

    def reset(self,):
        self.selected_vertiport_list:List = []

        pass
    
    def get_state(self,):

        return self.selected_vertiport_list, len(self.selected_vertiport_list)
    
    def set_state(self,):
        self.selected_vertiport_list  = [] # <- add vertiports to this list

    def get_possible_actions(self,):
        possible_vertiports = env.region.vertiports #TODO: env needs an attribute region.vertiports
        return possible_vertiports
        pass

    def is_terminal(self,):
        terminal = self.selected_vertiport_list == env.num_regions
        return terminal
        
    #! need to define a reward function and place in step
    def step(self, vertiport):
        self.selected_vertiport_list.append(vertiport)
        # run env-simulation with remaining region-vertiports selected randomly 
        # collect statistics 
        #TODO: this method does not return anything
        #TODO: fix this method or define new method that will return METRICS 
        metrics = env._collect_episode_end_metrics() 
        return metrics
    #! need to define a reward function and place in simulate_step
    def simulate_step(self,):
        #TODO: this method does not return anything
        #TODO: fix this method or define new method that will return METRICS 
        metrics = env._collect_episode_end_metrics()
        return metrics
        

    def get_action_space_size(self,):
        pass


mcts_env = VertiportDesignEnv()


# Hyperparameters for MCTS on Custom Grid World
NUM_SIMULATIONS = 100       # MCTS iterations per action selection (budget)
EXPLORATION_C = 1.414       # UCT exploration constant (sqrt(2) is common)
ROLLOUT_MAX_DEPTH = 50      # Max steps during the simulation phase
GAMMA_MCTS = 0.99           # Discount factor for rollout rewards

NUM_EPISODES_MCTS = 50      # Number of episodes to run the agent for visualization
MAX_STEPS_PER_EPISODE_MCTS = 200 # Max steps per episode




print(f"Starting MCTS Agent Interaction (Simulations per step={NUM_SIMULATIONS})...")

# --- MCTS Interaction Loop ---
mcts_run_rewards = []
mcts_run_lengths = []

for i_episode in range(1, NUM_EPISODES_MCTS + 1): #! Number of episodes to run the agent for visualization
    state: Tuple[int, int] = mcts_env.reset()
    episode_reward: float = 0.0
    #!                  for vertiport_env, state -> vertiport
    episode_path: List[Tuple[int, int]] = [state] # Store path for visualization
    
    for t in range(MAX_STEPS_PER_EPISODE_MCTS):   #! Max steps per episode
        if mcts_env.is_terminal(state):
            break # Already at goal
        
        # --- Use MCTS to choose the best action --- 
        root_node = mcts_search(state, 
                                NUM_SIMULATIONS,   #! MCTS iterations per action selection (budget)
                                EXPLORATION_C,     #! UCT exploration constant (sqrt(2) is common)
                                ROLLOUT_MAX_DEPTH, #! Max steps during the simulation phase
                                GAMMA_MCTS)        #! Discount factor for rollout rewards
        action = choose_best_mcts_action(root_node)
        # --- ------------------------------------ ---
        
        if action == -1: # Should not happen in this grid world unless goal reached
            print(f"Warning: MCTS returned invalid action (-1) at state {state}.")
            break

        # Take the chosen action in the real environment
        #!                                       action -> new vertiport to be added to state
        next_state, reward, done = mcts_env.step(action) # Use step() to advance env state
        #!                           INSIDE step -> that is adding the new vertiport through **action**, 
        #                                                                           i) randomly sample a vertiport from remaining regions 
        #                                                                          ii) now that all the vertiports are selected for the env
        #                                                                         iii) run MapEnv simulation to produce metrics, metrics = reward
        #! WHAT IS next_state ??
        #! WHAT IS done ??
        
        state = next_state
        episode_reward += reward
        episode_path.append(state)
        
        if done:
            break
            
    # --- End of Episode --- 
    mcts_run_rewards.append(episode_reward)
    mcts_run_lengths.append(t + 1)
    
    # Print progress
    if i_episode % 10 == 0:
        avg_reward = np.mean(mcts_run_rewards[-10:])
        avg_length = np.mean(mcts_run_lengths[-10:])
        print(f"Episode {i_episode}/{NUM_EPISODES_MCTS} | Avg Reward (last 10): {avg_reward:.2f} | Avg Length: {avg_length:.1f}")

print("MCTS Agent Interaction Finished.")