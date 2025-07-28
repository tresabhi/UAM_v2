from typing import List, Tuple, Dict, Optional
import numpy as np
from copy import deepcopy
import random
import time

from map_env_revised import MapEnv
from MCTS import mcts_search, choose_best_mcts_action


env = MapEnv(number_of_uav=0,
             num_ORCA_uav=0,
             number_of_vertiport=8,
             location_name='Austin, Texas, USA',
             airspace_tag_list=[],
             vertiport_tag_list=[('building', 'commercial')], #! use a tag-tag_str that has few vertiports  
             max_episode_steps=100,
             number_of_other_agents_for_model=7,#what is this???
             sleep_time=0,
             seed=70,
             obs_space_str='UAM_UAV',
             sorting_criteria='closest_first',
             render_mode=None,
             max_uavs=100,
             max_vertiports=150,
             make_uav_at_timestep=300,
             vp_design_problem=True)



class VertiportDesignEnv():
    def __init__(self,
                 env = env, #instance of MapEnv
                 map_env_timestep = 1000,
                 seed=123):
        self.seed = seed
        # create an instance of mapped_env
        self.env = env
        
        # run functions to setup env instance for vertiport design problem
        # this will create regions in airspace module 
        # NO vertiports
        self.env.set_airspace_vp_design()
        self.env.airspace.make_regions_dict('commercial', 4)


        self.map_env_timestep = map_env_timestep


        self.selected_vertiport_list:List = []

        self.start_state:Tuple = len(self.selected_vertiport_list) == 0
        self.goal_state = len(self.selected_vertiport_list) == self.env.airspace.num_regions #! remember this attr only available once env.airspace.make_region_and_vertiport_list() method is run
        
        self._internal_state:Tuple = self.selected_vertiport_list
        
        self.action_dim = None #! the dimension of action is supposed to change with every region
        
        return None

    def reset(self,):

        # self.selected_vertiport_list <- is state of vp_design_problem
        self.selected_vertiport_list:List = []
        return self.selected_vertiport_list
    
    def get_state(self,):
        '''get the internal state''' 
        return self.selected_vertiport_list
    
    def set_state(self, vertiport_list:List):
        '''manually set the internal state'''

        if len(self.selected_vertiport_list) == self.env.num_regions:
            raise ValueError('VertiportDesignEnv - set_state() - cannot assign vertiport to a complete/full vertiport_list')
        
        
        self.selected_vertiport_list += vertiport_list # <- add vertiports to this list
        
        return None

    def get_possible_actions(self, region):
        #! regions_dict will only be available once env.airspace.make_region_and_vertiport_list() method is run
        print(f'get vertiports from region: {region}')
        print(self.env.airspace.regions_dict.keys())
        possible_vertiports = self.env.airspace.regions_dict[region] #TODO: env needs an attribute region.vertiports
        self.action_dim = len(possible_vertiports)
        return possible_vertiports
        
    def is_terminal(self, state) -> bool:
        #! regions_dict will only be available once env.airspace.make_region_and_vertiport_list() method is run
        terminal = len(state) == self.env.airspace.num_regions
        return terminal
    
    #! need to define a reward function and place in step
    def step(self, vertiport_from_region):
        '''The selected vertiport (argument) will be added to the INTERNAL STATE(selecte_vertiport_list)
        Remaining vertiports will be added randomly for expansion and simulation.'''

        #self.selected_vertiport_list.append(vertiports_list_to_add_to_env)
        #! self.selected_vertiport_list is the internal state which is being mutated here 
        self.selected_vertiport_list.append(vertiport_from_region)

        vertiports_selected_by_mcts = deepcopy(self.selected_vertiport_list)

        vertiports_for_mapped_env = self.env.airspace.fill_vertiport_from_region(vertiports_selected_by_mcts)
        # step1 - add the vertiports list to env
        # run env-simulation with remaining region-vertiports selected randomly 
        self.env.airspace.set_vertiport_list_vp_design(vertiports_for_mapped_env)
        # step2 - reset env
        map_env_obs, map_env_info = self.env.reset(seed=self.seed)
        # step3 - collect pre-run env metrics 
        map_env_start_metric =  self.env._collect_initial_metrics()
        # step4 - run env for n-env steps (after n-env steps the mapped env will reach terminal state)
                                        #  and I will be able to collect end metrics 
        

        current_timestep = 0
        while current_timestep != self.map_env_timestep:
            auto_uav_action = self.env.agent.controller(map_env_info) #! convert env to self.env
            map_env_obs, reward, terminated, truncated, map_env_info = self.env.step(auto_uav_action) #! convert env to self.env
            current_timestep += 1
            if terminated or truncated:
                break
        

        
        # collect statistics 
        #TODO: this method does not return anything
        #TODO: fix this method or define new method that will return METRICS 
        map_env_end_metrics = self.env._collect_episode_end_metrics() 
        
        next_state = self.selected_vertiport_list
        reward = self._get_reward(map_env_start_metric, map_env_end_metrics)

        if len(self.selected_vertiport_list) == self.env.airspace.num_regions:
            done = True
        
        done = False

        return next_state, reward, done
    
    #! need to define a reward function and place in simulate_step
    #                       node.state   , action
    def simulate_step(self, current_state, vertiport_from_region):
        '''The vertiport argument is only used for evaluation/simulation
        It is not added to the internal state'''

        #self.selected_vertiport_list.append(vertiports_list_to_add_to_env)
        # print(f'in file vp_design_MCTS.simulate_step, printing current_state: {current_state}')
        # print(f'in file vp_design_MCTS.simulate_step, printing vertiport_from_region: {vertiport_from_region}')
        vertiports_selected_by_mcts = deepcopy(current_state)
        vertiports_selected_by_mcts.append(vertiport_from_region)
        # print(f'in file vp_design_MCTS.simulate_step, printing vertiport_from_region: {vertiports_selected_by_mcts}')
        vertiports_for_mapped_env = self.env.airspace.fill_vertiport_from_region(vertiports_selected_by_mcts)
        # step1 - add the vertiports list to env
        # run env-simulation with remaining region-vertiports selected randomly 
        self.env.airspace.set_vertiport_list_vp_design(vertiports_for_mapped_env)
        # step2 - reset env
        map_env_obs, map_env_info = self.env.reset(seed=self.seed)
        # step3 - collect pre-run env metrics 
        map_env_start_metric =  self.env._collect_initial_metrics()
        # step4 - run env for n-env steps (after n-env steps the mapped env will reach terminal state)
                                        #  and I will be able to collect end metrics 
        

        current_timestep = 0
        while current_timestep != self.map_env_timestep:
            auto_uav_action = self.env.agent.controller(map_env_info) #! convert env to self.env
            map_env_obs, reward, terminated, truncated, map_env_info = self.env.step(auto_uav_action) #! convert env to self.env
            current_timestep += 1
            if terminated or truncated:
                break
        

        
        # collect statistics 
        #TODO: this method does not return anything
        #TODO: fix this method or define new method that will return METRICS 
        map_env_end_metrics = self.env._collect_episode_end_metrics() 
        
        next_state = vertiports_selected_by_mcts
        reward = self._get_reward(map_env_start_metric, map_env_end_metrics)

        if len(vertiports_selected_by_mcts) == self.env.airspace.num_regions:
            done = True
        
        done = False

        return next_state, reward, done
        
    def get_action_space_size(self,region):
        # number of vertiport for a given region 
        return len(self.env.airspace.get_vertiports_of_region(region))
        
    def _get_reward(self, map_env_start_metrics, map_env_end_metrics):
        return random.random() * 10
        





# Hyperparameters for MCTS on Custom Grid World
NUM_SIMULATIONS = 1       # MCTS iterations per action selection (budget)
EXPLORATION_C = 1.414       # UCT exploration constant (sqrt(2) is common)
ROLLOUT_MAX_DEPTH = 5      # Max steps during the simulation phase
GAMMA_MCTS = 0.99           # Discount factor for rollout rewards

NUM_EPISODES_MCTS = 5      # Number of episodes to run the agent for visualization
MAX_STEPS_PER_EPISODE_MCTS = 2 # Max steps per episode




print(f"Starting MCTS Agent Interaction (Simulations per step={NUM_SIMULATIONS})...")

# create MCTS ENV
mcts_env = VertiportDesignEnv(env=env)

# --- MCTS Interaction Loop ---
mcts_run_rewards = []
mcts_run_lengths = []

for i_episode in range(1, NUM_EPISODES_MCTS + 1): #! Number of episodes to run the agent for visualization
    print(f'Episode: {i_episode}')
    # reset -> state is ()
    #! reset doesnt return anything
    state: List = mcts_env.reset() 
      
    episode_reward: float = 0.0
    #!                  for vertiport_env, state -> vertiport
    # episode_path = [(),]
    episode_path: List[List] = [state] # Store path for visualization
    
    for t in range(MAX_STEPS_PER_EPISODE_MCTS):   #! Max steps per episode
                               #state -> vp_design_env.selected_vertiport_list
        if mcts_env.is_terminal(state): #! mcts_env.is_terminal() doesn't take argument
            break # Already at goal
        
        # --- Use MCTS to choose the best action --- 
        root_node = mcts_search(state, 
                                mcts_env,
                                NUM_SIMULATIONS,   #! MCTS iterations per action selection (budget)
                                EXPLORATION_C,     #! UCT exploration constant (sqrt(2) is common)
                                ROLLOUT_MAX_DEPTH, #! Max steps during the simulation phase
                                GAMMA_MCTS)        #! Discount factor for rollout rewards
        action = choose_best_mcts_action(mcts_env, root_node)
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
        print(f'Current MCTS step: {t}')
        print(f'State: {state}')
        print('sleeping for 5 secs')
        time.sleep(5)
        
        if done:
            #! why is there no reset after terminal state, 
            #! if terminal state is reached before total_steps, 
            #! env should reset and continue again 
            #! WHY - break ???
            
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