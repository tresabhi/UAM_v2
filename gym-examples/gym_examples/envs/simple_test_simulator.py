import torch
from simple_env import SimpleEnv
from models.LSTM_A2C_core import LSTM_A2C




env = SimpleEnv(obs_space_str='seq', sorting_criteria='closest first')
obs, info= env.reset()
other_agent_state_size = len(env._get_obs()['other_agents_states'])
learning_agent_state_size = len(env._get_obs()) - 1
lstm_hidden_size = 64
action_size = env.action_space.shape[0]

# use a simple model from stable_baselines 
# use learning_agent state only with model from stable_baselines 
model = None #insert model from baselines  


#TODO: create method for agent state tensor -> use agent.get_state() and convert dict to tensor
learning_agent_states = torch.randn((9))
#TODO: 
other_agents_states = torch.from_numpy(env._get_obs()['other_agents_states']).unsqueeze(0).to(torch.float32)

# print(type(other_agents_states))

action, value, logp = model(learning_agent_states, other_agents_states)

print(action, value, logp)







