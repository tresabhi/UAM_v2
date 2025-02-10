import os 
import torch 
import pandas as pd
from torch.utils.data import Dataset


class CustomTrainingDataset(Dataset):
    '''
        This script will load data stored in directory. 
        The data is used for pretraining the embedding(RNN/GNN) and RL model.
    '''
    def __init__(self,data_file, transform=None, target_transform=None):
        self.data_frame = pd.read_csv(data_file) 
        self.transform = transform
        self.target_transform = target_transform
        

    def __len__(self):
        return len(self.data_frame)


    def __getitem__(self, idx):
        input_data = self.data_frame.iloc[idx, 0]
        output_data = self.data_frame.iloc[idx, 1]
        return input_data, output_data
        