from simple_env import SimpleEnv
from utils_data_transform import process_obs
from torch import tensor, unsqueeze




env = SimpleEnv(non_learning_uav_count=6,
                obs_space_str='seq', 
                sorting_criteria='closest first', 
                max_number_other_agents_observed=2) #this defines how many other_agents the learning agent is looking at 

(obs, info), ep_ret, ep_len = env.reset(), 0, 0


learing_agent_state, other_agents_states, mask = process_obs(obs)

learning_state_tensor = tensor(learing_agent_state)
other_agents_tensor = unsqueeze(tensor(other_agents_states), 0)


print(f'Type: {type(learning_state_tensor)}, shape: {learning_state_tensor.shape}')
print(f'Type: {type(other_agents_tensor)}, shape: {other_agents_tensor.shape}')


# print(f'obs: {obs}, \ninfo: {info}\nep_ret: {ep_ret}\nep_len: {ep_len}')
# print(obs)

# process_obs splits obs into two lists, learning_agent, other_agents_states
# learning agent holds mask data
# print(f'model obs: {process_obs(obs)}')
# learning_state_shape = env.get_obs_shape()['learning_agent_state_shape']
# print(f'learning agent obs shape: {learning_state_shape}')

# print(process_obs(obs)[0]) 
# print()
# print(process_obs(obs)[1])


# print(env.observation_space['other_agent_state'].shape)
# print(env.action_space.shape)
# print(env.max_number_other_agents_observed)
# print(env.get_obs_shape()['learning_agent_state_shape'], env.get_obs_shape()['other_agents_states_shape'])

# for i in range(10):
#     sample_action = 100 * env.action_space.sample()
#     print(f'sample action: {sample_action}')
    
#     env.agent.dynamics.update(env.agent, sample_action)
#     current_agent_state = env.agent.get_state()

#     print(f'Agent state: {current_agent_state}')