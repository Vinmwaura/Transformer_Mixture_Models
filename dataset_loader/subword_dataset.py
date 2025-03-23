import math
import random

import torch
from torch.utils.data import Dataset

"""
Load SubWord Dataset.
"""
class SubWord_Dataset(Dataset):
    def __init__(
            self,
            dataset,
            context_window=200):
        self.dataset = dataset
        self.context_window = context_window
    
    def __len__(self):
        return len(self.dataset) - (self.context_window + 1)

    def __getitem__(self, index):
        input_data = self.dataset[index:index + self.context_window]
        target_data = self.dataset[index + 1:index + self.context_window + 1]

        input_tensor = torch.tensor(input_data)
        target_tensor = torch.tensor(target_data)

        return input_tensor, target_tensor    
