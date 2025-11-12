import torch
from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval

class SentimentDataset(Dataset):
    def __init__(self, filename):
        self.df = pd.read_csv(filename, converters={'input_x': literal_eval})
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, ind):
        input_x = torch.tensor(self.df.loc[ind, 'input_x'])
        label = torch.tensor(self.df.loc[ind, 'Label'],dtype=torch.float)
        return input_x, label