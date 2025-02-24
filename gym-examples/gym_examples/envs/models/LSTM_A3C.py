import torch
from torch import nn


# This model will accept an array
# where each datapoint contains data about one other_UAV(intruder)
# LSTM works on sequence of data, each datapoint will be passed,
# according to the sequence its stored in the array.
# The LSTM will output one vector.  


class LSTM_embedding_Network(nn.Module):
    def __init__(self, input_size, hidden_size):
        # define all the layers needed for LSTM embedding
        super().__init__()
        # input 
        # LSTM layer
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first = True)

    # x: np.array - shape(no_other_agents, feats):(4,7)or(10,7)
    # if batched input input shape - N,L,H 
    # N:batch size
    # L:sequence length 
    # H_in: input size 
    def forward(self, x:torch.Tensor):
        # INPUT - dtype - torch.Tensor
        # define the flow of input data x
        # OUTPUT - a vector, dtype - torch.Tensor
        h0 = torch.zeros((1, self.hidden_size))
        c0 = torch.zeros((1, self.hidden_size))
        #! QUESTION 
        #! 
        output, (hn,cn) = self.lstm(x, (h0, c0))
        return output, hn, cn


    # some points to think about, 
    # 1) is there a way to test this model by itself to make sure its working as intended
    # 2) for ablation study, do we need to perform ablation study for this work
    #   if yes
    #           how to perform ablation study for LSTM-RL   
if __name__ == '__main__':
    lstm_net = LSTM_embedding_Network(7,6)
    x = torch.rand((4,7)) #other state vector 
    
    out, hn,cn = lstm_net(x)


    # create input vector for RL ALGO 
    rl_in = [uav_state, hn]
    # insert RL ALGO 

    # use RL_ALGO and pass input vector to it 
    pred_action = RL_algo(rl_in)

    # define loss function for action
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(pred_action, action)

    #perform back prop on the whole model, RL_ALGO+LSTM
    loss.backward()

    print(out)
    print()
    print(hn)