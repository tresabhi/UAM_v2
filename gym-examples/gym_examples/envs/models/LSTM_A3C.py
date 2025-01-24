import torch
from torch import nn


# This model will accept an array
# where each datapoint contains data about one other_UAV(intruder)
# LSTM works on sequence of data, each datapoint will be passed,
# according to the sequence its stored in the array.
# The LSTM will output one vector.  


class LSTM_embedding_Network(nn.Module):
    def __init__(self):
        # define all the layers needed for LSTM embedding
        super().__init__()
        pass

    def forward(self, x:torch.Tensor):
        # INPUT - dtype - torch.Tensor
        # define the flow of input data x
        # OUTPUT - a vector, dtype - torch.Tensor
        pass


    # some points to think about, 
    # 1) is there a way to test this model by itself to make sure its working as intended
    # 2) for ablation study, do we need to perform ablation study for this work
    #   if yes
    #           how to perform ablation study for LSTM-RL   
