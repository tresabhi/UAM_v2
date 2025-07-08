import math
import random
from typing import List, Tuple, Dict, Optional









class MCTSNode:
    """ Represents a node in the Monte Carlo Tree Search. """
    #                  state: of env,          parent: MCTSNode with initial state where the selected_vertiport_list is empty
    def __init__(self, state: Tuple[int, int], parent: Optional['MCTSNode'] = None, action_that_led_here: Optional[int] = None):
        self.state: Tuple[int, int] = state #! this the state of ENV/SIM,
        #                                      the list of selected_vertiports so far
        #                                      for each node this list will be different,
        #                                      since each node will contain different number of regions,
        #                                      ex: start_node has empty list, level 3 will have 3 VERTIPORTS ... and so on  
        self.parent: Optional[MCTSNode] = parent

        #! action_that_led_here -> is the veriport choice, that makes current_state be the vertiport from the region
        self.action_that_led_here: Optional[int] = action_that_led_here # Action taken by parent to reach this node
        
        self.children: Dict[int, MCTSNode] = {} # Maps action(the act of choosing the vertiport) -> child node(all possible vertiports)
        
        #TODO: add attr -> env.airspace.regions.vertiports OR env.regions.vertiports
        # untried_actions are related to current REGION(level of the tree) -  region.vertiports
        #!                                self.mcts_env -> this is NOT defined in the __init__
        self.untried_actions: List[int] = self.mcts_env.get_possible_actions(state) # Actions not yet expanded from this node
        # logic for accomplishing the above
        # regions are ordered with numbers - for a given vertiport from region n 
        #                           possible actions are vertiports from region n+1 
        # since a node is a vertiport (and its region information)
        # i can use these two to query the next region and all available vertiports there
        # copy/deepcopy the vertiports from the following region into *untried_actions*
        #                                                              each of the vertiports from region n+1 wil be tried once and untried_actions will become an empty list 
        
        self.visit_count: int = 0   # remember for vertiport design problem this means a list of vertiports that have been used/selected
        self.total_value: float = 0.0 # Sum of rewards from rollouts through this node
        
    def is_fully_expanded(self) -> bool:
        """ Checks if all possible actions from this node have been expanded. """
        return len(self.untried_actions) == 0
        
    def is_terminal(self) -> bool:
        """ Checks if the state represented by this node is terminal. """
        #!     self.mcts_env -> this is NOT defined in the __init__
        # in vertiport design problem - the env has reached terminal state 
        # when list of selected vertiports has one vertiport from each region 
        return self.mcts_env.is_terminal(self.state)
        
    def get_average_value(self) -> float:
        """ Calculates the average value Q(s) for this node. """
        if self.visit_count == 0:
            return 0.0 # Or perhaps -infinity or another default for unvisited
        return self.total_value / self.visit_count
    

def select_best_child_uct(node: MCTSNode, exploration_constant: float) -> MCTSNode:
    """
    Selects the child node with the highest UCT score.
    UCT = Q(child) + C * sqrt(ln(N(parent)) / N(child))
    
    Parameters:
    - node (MCTSNode): The parent node from which to select.
    - exploration_constant (float): The constant C balancing exploration/exploitation.
    
    Returns:
    - MCTSNode: The child node with the highest UCT value.
    """
    best_score = -float('inf')
    best_child = None
    #! (starting state)empty_list.children points to region 1 dict
    for action, child in     node.children.items():
        if child.visit_count == 0:
            # Ensure unvisited children are selected first
            uct_score = float('inf') 
        else:
            exploit_term = child.get_average_value()
            explore_term = exploration_constant * math.sqrt(
                math.log(node.visit_count) / child.visit_count
            )
            uct_score = exploit_term + explore_term
            
        if uct_score > best_score:
            best_score = uct_score
            best_child = child
            
    if best_child is None:
        # This should ideally not happen if the node has children
        raise RuntimeError("Selection failed: No children found or error in UCT calculation.")

    return best_child


def expand_node(mcts_env, node: MCTSNode) -> MCTSNode:
    #! EXPANSION is about trying UNTRIED actions 
    """
    Expands the given node by choosing an untried action, simulating it,
    and adding the resulting state as a new child node.

    Parameters:
    - node (MCTSNode): The leaf node to expand.

    Returns:
    - MCTSNode: The newly created child node.
    """
    #! since node.untried_actions are being 'pop'ed once all the 'actions' aka vertiports are tried the untried_actions list will be empty
    #! expand_node() will not be called on a node whose actions have all been tried 
    if not node.untried_actions:
        raise RuntimeError("Cannot expand a fully expanded node.")
        
    # Choose an action to expand (e.g., the first untried one)
    action = node.untried_actions.pop() #! <- this method should return a vertiport
    
    # Simulate this action from the node's state using the environment model
    #!   what should this       simulate_step() DO!!!!!
    #! next_state <- what is the meaning of this next_state in terms of MCTS/vertiport design problem 
    next_state, _, _ = mcts_env.simulate_step(node.state, action) # We only need next state here
    
    # Create the new child node
    child_node = MCTSNode(state=next_state, parent=node, action_that_led_here=action)
    #* since action selected a vertiport from the next region, v_k_R3
    #* child node is [v_i_R1, v_j_R2, v_k_R3, ......] 
    
    # Add the child to the parent's children dictionary
    node.children[action] = child_node
    
    return child_node


#!                          how to define max_depth
#                                         max_depth = num_regions ??  
def perform_rollout(mcts_env, start_node: MCTSNode, max_depth: int, gamma: float) -> float:
    """
    Performs a Monte Carlo simulation (rollout) from the start node's state.
    Uses a random policy for the rollout.

    Parameters:
    - start_node (MCTSNode): The node from which to start the simulation.
    - max_depth (int): Maximum number of steps for the rollout.
    - gamma (float): Discount factor for the rollout rewards.

    Returns:
    - float: The total discounted reward obtained during the rollout.
    """
    current_state = start_node.state
    total_discounted_reward: float = 0.0
    current_discount: float = 1.0
    depth = 0

    while not mcts_env.is_terminal(current_state) and depth < max_depth:
        # Choose a random action (rollout policy)
        possible_actions = mcts_env.get_possible_actions(current_state)
        if not possible_actions: # Should not happen if not terminal, but safe check
            break 
        action = random.choice(possible_actions) #! this random.choice will select a vertiport from a region at random
        
        # Simulate the step using the environment model
        next_state, reward, done = mcts_env.simulate_step(current_state, action)
        
        total_discounted_reward += current_discount * reward
        
        current_state = next_state
        current_discount *= gamma
        depth += 1
        
    return total_discounted_reward

def backpropagate(node: MCTSNode, reward: float) -> None:
    """
    Updates the visit counts and total values of nodes up the tree.

    Parameters:
    - node (MCTSNode): The node from which the simulation started.
    - reward (float): The result (total discounted reward) of the simulation.
    """
    current_node: Optional[MCTSNode] = node
    while current_node is not None: #! root_node needs to point to None
        current_node.visit_count += 1
        current_node.total_value += reward 
        # Move up to the parent node
        current_node = current_node.parent



def mcts_search(
    root_state: Tuple[int, int],
    num_simulations: int,
    exploration_constant: float,
    rollout_max_depth: int,
    gamma: float
) -> MCTSNode:
    """
    Performs the Monte Carlo Tree Search (MCTS) process for a given number of simulations.

    Parameters:
    - root_state (Tuple[int, int]): The starting state for the search.
    - num_simulations (int): The number of simulations to perform.
    - exploration_constant (float): The exploration constant (C) used in the UCT formula.
    - rollout_max_depth (int): The maximum depth for rollouts during the simulation phase.
    - gamma (float): The discount factor for rewards during rollouts.

    Returns:
    - MCTSNode: The root node of the search tree after simulations.
    """
    # Create the root node for the current state
    root_node: MCTSNode = MCTSNode(state=root_state)

    # Perform the specified number of simulations
    for _ in range(num_simulations):
        current_node: MCTSNode = root_node

        # --- 1. Selection ---
        # Traverse down the tree using UCT until a leaf node is found
        while not current_node.is_terminal() and current_node.is_fully_expanded() and current_node.children:
            # try all untried nodes and then choose the node that has highest UCT value 
            #current_node (new_node) a child of current_node from above  
            current_node = select_best_child_uct(current_node, exploration_constant)

        # --- 2. Expansion ---
        # Expand the current node if it is not terminal and not fully expanded
        simulation_start_node: MCTSNode = current_node
        if not current_node.is_terminal() and not current_node.is_fully_expanded():
            simulation_start_node = expand_node(current_node)

        # --- 3. Simulation ---
        # Perform a rollout from the expanded node and calculate the reward
        rollout_reward: float = perform_rollout(simulation_start_node, rollout_max_depth, gamma)

        # --- 4. Backpropagation ---
        # Update the statistics of all nodes along the path to the root
        backpropagate(simulation_start_node, rollout_reward)

    # Return the root node with updated statistics
    return root_node




def choose_best_mcts_action(mcts_env, root_node: MCTSNode) -> int:
    """
    Selects the best action from the root node after MCTS is complete.
    Typically chooses the action leading to the most visited child node.

    Parameters:
    - root_node (MCTSNode): The root of the search tree after simulations.

    Returns:
    - int: The best action to take.
    """
    best_visit_count = -1
    best_action = -1

    if not root_node.children:  # If no actions were expanded (e.g., only 1 simulation)
        # Fallback: Choose a random action if possible
        possible_actions = mcts_env.get_possible_actions(root_node.state)
        if possible_actions:
            return random.choice(possible_actions)
        else:
            return -1  # Or handle error

    # Find the action leading to the child with the highest visit count
    for action, child in root_node.children.items():
        if child.visit_count > best_visit_count:
            best_visit_count = child.visit_count
            best_action = action

    if best_action == -1:
        # Fallback if all children have 0 visits (shouldn't happen with proper MCTS)
        possible_actions = list(root_node.children.keys())
        if possible_actions:
            return random.choice(possible_actions)
        else:
            return -1

    return best_action