import os 
import torch 
from torch.utils.data import Dataset


class CustomTrainingDataset(Dataset):
    '''
        This script will load data stored in directory. 
        The data is used for pretraining the embedding(RNN/GNN) and RL model.
    '''
    def __init__(self,input, output, transform=None, target_transform=None):
        # provide the location of input data
        # provide the location of output data
        pass

    def __len__(self):
        pass

    def __getitem__(self): # DOES THIS METHOD ONLY ACCEPT ONE ARGUMENT - idx
                           # can this method accept more than one argument
        pass