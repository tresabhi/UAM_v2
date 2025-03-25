import torch
import time

from models.LSTM_A2C_core import LSTM_A2C
from models.LSTM_A2C_utils import VPGBuffer, update_actor_critic
from models.utils.logx import Logger, EpochLogger
# logger to log obs, action, adv, rew, ret, value, log_p, 
# store gamma and lambda 
# pointer, path_startindex, and max_size of logger/buffer 
# 

from simple_env import SimpleEnv

env = SimpleEnv()
epochs = None
steps_per_epoch = None
max_ep_len = None
save_freq = None

#! need to provide arguments 
ac = LSTM_A2C()
buf = VPGBuffer()
logger = EpochLogger()

start_time = time.time()


#### ENV RESET ####
#! RESET method needs fix
obs, ep_ret, ep_len = env.reset(), 0, 0
####          ####


# Main loop: collect experience in env and update/log each epoch
for epoch in range(epochs):
    for t in steps_per_epoch:
        
        #### LSTM-A2C MODEL STEP ####
        # get action, and value from actor-critic model 
        a, v, logp = ac.step(torch.as_tensor(obs, dtype=torch.float32))
        ####                    ####
        
        
        #### ENV STEP ####
        #! STEP method needs fix
        next_o, r, d, _ = env.step(a)
        ####          ####

        ep_ret += r
        ep_len += 1

        # save and log
        buf.store(obs, a, r, v, logp)
        logger.store(VVals=v)
        
        # Update obs (critical!)
        obs = next_o

        timeout = ep_len == max_ep_len
        terminal = d or timeout
        epoch_ended = t==steps_per_epoch-1

        if terminal or epoch_ended:
            if epoch_ended and not(terminal):
                print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
            # if trajectory didn't reach terminal state, bootstrap value target
            if timeout or epoch_ended:
                _, v, _ = ac.step(torch.as_tensor(obs, dtype=torch.float32))
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