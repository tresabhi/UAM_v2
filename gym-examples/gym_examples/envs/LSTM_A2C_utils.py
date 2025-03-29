import numpy as np
import torch
from torch.optim import Adam
import gymnasium 
import time

from LSTM_A2C_core import LSTM_A2C as core

from mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar

#! COST FUNCTION - VALUE NETWORK:
#! cost_v = 0.5 * tf.reduce_sum(tf.square(self.y_r - self.logits_v), axis=0) 


#### TESTING MPI ####
x = np.array([1,2,3])
mean, std = mpi_statistics_scalar(x)
print(mean, std)
####    END    ####

class VPGBuffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """
                                    # what is gamma, and lam
                                    # gamma - discount factor, lambda ???
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off #! for one 'RUN' this method gets called everytime an episode ends, 
        by an epoch ending. This looks back in the buffer to where the #!  or 'RUN' reaches end of epoch counter.
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda, #! for my use case I cannot go till the end of episode/epoch
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val) # environment reward
        vals = np.append(self.val_buf[path_slice], last_val) # value from value_function, v(s)
        
        # the next two lines implement GAE-Lambda advantage calculation
            #! i dont understand this 
            #! what is deltas
            # this appers to be (r + v(s_t+1)) - v(s_t)
            # so what is gamma - Gamma is the discount factor
            # and why are they called deltas - I figured this out, TD error, aka delta, I think TD error would have been more in line with RL terminology  
    #*  deltas AKA TD_Error
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1] # r(t+1) + v(s+1) - v(s), v(s)= E(R)
            # what is lambda ???
            # g = grad_log_pi * (RTG - A)
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam) 
        
        # the next line computes rewards-to-go(RTG), to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        #! find a method that will calculate the mean and std_dev for a ndarray data
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, 
                    act=self.act_buf, 
                    ret=self.ret_buf,   # Reward to Go
                    adv=self.adv_buf,   # Advantage function
                    logp=self.logp_buf) # lop_probability of action
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()} # numpy to tensor conversion







def compute_loss_pi(data, ac):
    obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

    # Policy loss
        # pi is distribution - in this case its a Normal distribution
    pi, logp = ac.pi(obs, act)
    loss_pi = -(logp * adv).mean() # REMEMBER:
                                    # grad_J approximated - grad_log_pi,
                                    # the negative because, pytorch's gradient op always minimizes loss
                                    # since we want to maximise the  policy performance, we use negative operation
                                    # the mean because its calculating loss on a batch 

    # Useful extra info
        #! why are these useful ???
    approx_kl = (logp_old - logp).mean().item()
    ent = pi.entropy().mean().item()
    pi_info = dict(kl=approx_kl, ent=ent)

    return loss_pi, pi_info



# Set up function for computing value loss
def compute_loss_v(data, ac):
    obs, ret = data['obs'], data['ret']
    return ((ac.v(obs) - ret)**2).mean() # MSE loss 


def pi_optimizer(optim_str, ac, pi_lr):
    '''ADAM or RMS_Prop.
    adam: for ADAM optimizer,
    rms:  for RMS_Prop optimizer'''
    if optim_str == 'adam':
        return Adam(ac.pi.parameters(), lr = pi_lr)
    elif optim_str == 'rms':
        pass
    else:
        raise RuntimeError('Invalid str for policy optimizer choice')


def vf_optimizer(optim_str, ac, v_lr):
    '''ADAM or RMS_Prop
    adam: for ADAM optimizer,
    rms:  for RMS_Prop optimizer'''
    if optim_str == 'adam':
        return Adam(ac.v.parameters(), lr = v_lr)
    elif optim_str == 'rms':
        pass
    else:
        raise RuntimeError('Invalid str for value optimizer choice')


# call to this method will update the policy network once, 
# and update the value network train_v_iters times.
# During training this method will need to be called iteratively.  
def update_actor_critic(buf, logger,train_v_iters):
        '''
        Use collected data to update the parameters of pi and v network.
        This method updates policy network once and value network train_v_iters times. 
        '''
        # data for whole epoch
        data = buf.get()

        # Get loss and info values before update
        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with a single step of gradient descent
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()

        #mpi_avg_grads(ac.pi)    # average grads across MPI processes

        pi_optimizer.step()

        # Value function learning
        for i in range(train_v_iters): 
            # REMEMBER: train_v_iters = 80,
            # meaning using the same loss its taking 80 update steps for the val_func
            #! WHY take so many gradient update steps  ???
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            
            #mpi_avg_grads(ac.v)    # average grads across MPI processes
            
            vf_optimizer.step()

        # Log changes from update
        kl, ent = pi_info['kl'], pi_info_old['ent']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))



def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)



def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]