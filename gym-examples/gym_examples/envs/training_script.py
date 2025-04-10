from torch import tensor, unsqueeze
import torch
import time

from LSTM_A2C_core import LSTM_A2C
from LSTM_A2C_utils import VPGBuffer, update_actor_critic, count_vars
from logx import Logger, EpochLogger
from utils_data_transform import process_obs
from mpi_pytorch import sync_params
# logger to log obs, action, adv, rew, ret, value, log_p, 
# store gamma and lambda 
# pointer, path_startindex, and max_size of logger/buffer 
# 

from simple_env import SimpleEnv


#### model HYPER-PARAMETER ####
env = SimpleEnv(obs_space_str='seq', sorting_criteria='closest first')
epochs = 100               # for training the actor and critic network
steps_per_epoch = 4000      # data collection steps - each step collects state, action, next_state
                            # steps per epoch - should be more than episode length                               
max_ep_len = 3500           # gym episode length
save_freq = 100


#! need to provide arguments 
other_agents_states_size = env.get_obs_shape()['other_agents_states_shape']
learning_agent_state_size = env.get_obs_shape()['learning_agent_state_shape'] 
lstm_hidden_size = 64
fc_output_size = 256
action_size = env.action_space.shape[0]
lstm_a2c = LSTM_A2C(other_agent_states_size=other_agents_states_size, 
                    learning_agent_state_size=learning_agent_state_size,
                    lstm_hidden_size=lstm_hidden_size,
                    FC_output_size=fc_output_size, 
                    action_size=action_size) 

print(f'learning agent state size: {learning_agent_state_size}')

#### ENV RESET ####
(obs, info), ep_ret, ep_len = env.reset(), 0, 0 
print(f'observation: {obs}')
####          ####

buf = VPGBuffer(learning_agent_state_size, action_size, steps_per_epoch*epochs) 
# setup logger and save config
logger = EpochLogger()
# print(locals())
# logger.save_config(locals())

# Sync params across process
#! read doc to see if I need this 
# sync_params(lstm_a2c)

# Count variables
var_counts = tuple(count_vars(module) for module in [lstm_a2c.a2c.pi, lstm_a2c.a2c.v])
logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)


start_time = time.time() #! why do i need this - its being used in logger 




# Main loop: collect experience in env and update/log each epoch
for epoch in range(epochs):
    for t in range(steps_per_epoch):
        #### LSTM-A2C MODEL STEP ####
        # get action, and value from actor-critic model 
        print(f'step: {t}')
        learning_state, other_agents_states, mask = process_obs(obs)
        learning_state_tensor = tensor(learning_state, dtype=torch.float32)
        other_agents_tensor = unsqueeze(tensor(other_agents_states, dtype=torch.float32), 0)
        
        #FIX: remove torch.as_tensor, convert to torch tensor inside model
        a, v, logp = lstm_a2c(learning_state_tensor, other_agents_tensor)
    
        #### ENV STEP ####
        #! STEP method needs fix
        next_obs, reward, done, _ = env.step(a)
        
        ep_ret += reward
        ep_len += 1

        # save and log
        buf_obs = learning_state
        buf.store(buf_obs, a, reward, v, logp)
        logger.store(VVals=v)
        
        # Update obs (critical!)
        obs = next_obs

        timeout = ep_len == max_ep_len
        terminal = done or timeout
        epoch_ended = t==steps_per_epoch-1

        if terminal or epoch_ended:
            if epoch_ended and not(terminal):
                print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
            # if trajectory didn't reach terminal state, bootstrap value target
            if timeout or epoch_ended:
                learning_state, other_agents_states, mask = process_obs(obs) 
                learning_state_tensor = tensor(learning_state, dtype=torch.float32)
                other_agents_tensor = unsqueeze(tensor(other_agents_states, dtype=torch.float32), 0)
                _, v, _ = lstm_a2c(learning_state_tensor, other_agents_tensor)
            else:
                v = 0
            buf.finish_path(v)
            if terminal:
                # only save EpRet / EpLen if trajectory finished
                logger.store(EpRet=ep_ret, EpLen=ep_len)
            obs, ep_ret, ep_len = env.reset(), 0, 0


    # Save model
    if (epoch % save_freq == 0) or (epoch == epochs-1):
        logger.save_state({'env': env}, None)

    # Perform VPG update!
    update_actor_critic()

    # Log info about epoch
    logger.log_tabular('Epoch', epoch)
    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.log_tabular('VVals', with_min_and_max=True)
    logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
    logger.log_tabular('LossPi', average_only=True)
    logger.log_tabular('LossV', average_only=True)
    logger.log_tabular('DeltaLossPi', average_only=True)
    logger.log_tabular('DeltaLossV', average_only=True)
    logger.log_tabular('Entropy', average_only=True)
    logger.log_tabular('KL', average_only=True)
    logger.log_tabular('Time', time.time()-start_time)
    logger.dump_tabular()