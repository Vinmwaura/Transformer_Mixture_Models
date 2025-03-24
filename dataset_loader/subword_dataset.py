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
            tr_data,
            delimiter,
            start_token,
            padding_token,
            context_window=200):
        self.start_token = start_token

        self.dataset = []

        # Init append chunk from start of dataset.
        self.dataset.append(tr_data[0:0 + context_window ])

        # Iterate over dataset and create chunks using delimiter as point to start.
        for index in range(1, len(tr_data) - context_window):
            if tr_data[index] == delimiter and tr_data[index + 1] != delimiter:
                temp_chunk_data = tr_data[index + 1:index + 1 + context_window]

                len_data = len(temp_chunk_data)
                temp_padding_list = [padding_token] * (len_data - context_window)
                temp_chunk_data.extend(temp_padding_list)
                self.dataset.append(temp_chunk_data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input_data = [self.start_token] + self.dataset[index][:-1]
        target_data = self.dataset[index][:]

        input_tensor = torch.tensor(input_data)
        target_tensor = torch.tensor(target_data)

        return input_tensor, target_tensor    
