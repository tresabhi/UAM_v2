import numpy as np
import scipy.signal
from gymnasium.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

# This model will accept an array
# where each datapoint contains data about one other_UAV(intruder)
# LSTM works on sequence of data, each datapoint will be passed,
# according to the sequence its stored in the array.
# The LSTM will output one vector.  


class LSTM_embedding_Network(nn.Module):
    def __init__(self, feature_size, hidden_size, batch_size=1):
        super().__init__()
        # input 
        # LSTM layer
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lstm = nn.LSTM(feature_size, hidden_size, batch_first = True)

    # x: np.array - shape(no_other_agents, feats):(4,7)or(10,7)
    # if batched input input shape - N,L,H 
    # N:batch size
    # L:sequence length 
    # H_in: input size 
    def forward(self, x:torch.Tensor):
        # INPUT - dtype - torch.Tensor
        # define the flow of input data x
        # OUTPUT - a vector, dtype - torch.Tensor
        h0 = torch.zeros((1, self.batch_size, self.hidden_size))
        c0 = torch.zeros((1, self.batch_size, self.hidden_size))
        #! QUESTION 
        #! 
        output, (hn,cn) = self.lstm(x, (h0, c0))
        return output, (hn, cn)


#TODO: Add regularization and normalization 




class FC(nn.Module):
    def __init__(self, combined_state_input_size, output_size):
        super().__init__()
        self.FC_layer = nn.Sequential(nn.Linear(combined_state_input_size, output_size),
                                      nn.ReLU(),
                                      nn.Linear(output_size, output_size),
                                      nn.ReLU())
    def forward(self, input):
        return self.FC_layer(input)

# parent class for discrete actor, and continuous actor
class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

#TODO: add to_device, regularization, normalization
class DiscreteActor(Actor):
    def __init__(self, obs_size, discrete_action_size):
        super().__init__()
        self.logits_net = nn.Linear(obs_size, discrete_action_size) #check rl_collision implementation - NetworkVPCore, no activation
    
    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return torch.distributions.Categorical(logits=logits)
    
    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)
    
class GausiannActor(Actor):
    def __init__(self, obs_size, continuous_action_size):
        super().__init__()
        log_std = -0.5 * np.ones(continuous_action_size, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = nn.Sequential(nn.Linear(obs_size, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, continuous_action_size))
    
    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution
        # REASON FOR SUMMING THE LOG_PROBS: the action is a multivariate random variables which 
        #                                   is modeled as distinct Normal random variable,
        #                                   so to find the probability of the multivariate normal rv,
        #                                   we need to multiply the distinct probabilities, or in this case 
        #                                   sum of log probabilities. Log operation changes multiplication to sum.
        
#TODO: rename obs_size -> obs_shape, and action_size to action_shape
class Critic(nn.Module):
    def __init__(self, obs_size):
        super().__init__()
        self.critic = nn.Sequential(nn.Linear(obs_size,256),
                                    nn.ReLU(),
                                    nn.Linear(256, 1))

    def forward(self, obs):
        return torch.squeeze(self.critic(obs), -1) #TODO: check its output to ensure it returns a scalar/vector with 1 element 

class ActorCritic(nn.Module):
    #TODO: NEED TO COMPLETE THIS IMPLEMENTATION 
    def __init__(self, observation_space, action_space):
        super().__init__()
        obs_dim = observation_space.shape[0] #TODO: why shape[0], current implementation is a copy from spinup, check with UAM req

        if isinstance(action_space, Box):
            self.pi = GausiannActor(obs_size=obs_dim, continuous_action_size=action_space.shape[0])
        elif isinstance(action_space, Discrete):
            self.pi = DiscreteActor(obs_size=obs_dim, discrete_action_size=action_space.shape[0])

        self.v = Critic(obs_size=obs_dim)
    
    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
    


class ContinuousActorCritic(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.pi = GausiannActor(obs_size=obs_size, continuous_action_size=action_size)
        self.v = Critic(obs_size=obs_size)
    def step(self, obs):
            with torch.no_grad():
                # distribution
                pi = self.pi._distribution(obs)
                # action sample from distribution
                a = pi.sample()
                # lop_probability of action,a using distribution pi
                logp_a = self.pi._log_prob_from_distribution(pi, a)
                # value from value network 
                v = self.v(obs)
            return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        '''Returns JUST the action a from step method defined above which returns 3 items in tuple'''
        return self.step(obs)[0] 

class LSTM_A2C(nn.Module):
    def __init__(self, other_agent_states_size, learning_agent_state_size, lstm_hidden_size, FC_output_size, action_size):
        super().__init__()
        self.lstm_net = LSTM_embedding_Network(other_agent_states_size, lstm_hidden_size)
        self.fc_network = FC(combined_state_input_size= lstm_hidden_size + learning_agent_state_size, output_size=FC_output_size)
        self.a2c = ContinuousActorCritic(obs_size= FC_output_size, action_size=action_size)
    
    def forward(self, other_agent_states, learning_agent_state):
        # LSTM out dim: 64
        lstm_out, (h_out, c_out) = self.lstm_net(other_agent_states)
        # combined dim: 64 + 9 = 73
        combined_state = torch.cat((learning_agent_state, torch.squeeze(h_out)))
        # fc out dim: 256
        fc_out = self.fc_network(combined_state)
        # action dim: , value dim: 1
        action, value, logp = self.a2c.step(fc_out)

        return action, value, logp


    
    # some points to think about, 
    # 1) is there a way to test this model by itself to make sure its working as intended
    # 2) for ablation study, do we need to perform ablation study for this work
    #   if yes
    #           how to perform ablation study for LSTM-RL   
if __name__ == '__main__':
    # other agent state: 7 [other_agent_x, y, vx, vx, distance_to_other_agent, other_agent_size, combined_size]
    other_agent_state_size = 7
    # learning_agent_state: 9 []
    learning_agent_state_size = 9

    lstm_hidden_size = 64
    FC_output_size = 256#check gym_collision paper
    action_size = 2 # or use actions list from MIT_ACL for discrete action 
    value_size = 1
    
    # Learning model 
    lstm_a2c = LSTM_A2C(other_agent_states_size=other_agent_state_size,
                        learning_agent_state_size=learning_agent_state_size,
                        lstm_hidden_size=lstm_hidden_size, 
                        FC_output_size=FC_output_size,
                        action_size=action_size
                        )
    

    print(lstm_a2c)
    
    ### TESTING MODEL ### 
    # batch: 1, other_agents:4, other_agent_state:7
    other_agent_states = torch.randn((1,4,7))
    learning_agent_state = torch.randn((9))

    action, value, logp = lstm_a2c(other_agent_states, learning_agent_state)
    print(action, value, logp)

    


    #### LSTM test START ####
    
    # x = torch.rand((4,7)) #other state vector 
    
    # out, hn,cn = lstm_net(x)


    # # create input vector for RL ALGO 
    # rl_in = [uav_state, hn]
    # # insert RL ALGO 

    # # use RL_ALGO and pass input vector to it 
    # pred_action = RL_algo(rl_in)

    # # define loss function for action
    # loss_fn = torch.nn.MSELoss()
    # loss = loss_fn(pred_action, action)

    # #perform back prop on the whole model, RL_ALGO+LSTM
    # loss.backward()

    # print(out)
    # print()
    # print(hn)

    #### LSTM test END ####