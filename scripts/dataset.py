import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as Data



class SentimentDataset(Data.Dataset):
    def __init__(self, data_array) -> None:
        super(SentimentDataset, self).__init__()
        
        self.data_array = data_array
        input_ids = torch.tensor(self.data_array[:, 1:])
        self.labels = torch.tensor(self.data_array[:, 0], dtype=torch.int64)
        self.length = len(self.labels)
        # self.labels = F.one_hot(torch.tensor())
        attention_mask = (input_ids != 0)


        nonzero_index = torch.zeros((self.length, 1), dtype=torch.int64)

        for i in range(self.length):
            x = input_ids[i]
            x_zero = torch.nonzero(x)
            nonzero_index[i, 0] = x_zero[-1]
            input_ids[i, :x_zero[-1] + 1] -=1
        self.nonzero_index = nonzero_index
        self.data = input_ids
        print("data shape: {}".format(self.data.shape))
        self.attention_mask = attention_mask
        
        print("Build dataset successfully!")

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.data[idx], self.attention_mask[idx], self.labels[idx], self.nonzero_index[idx]

if __name__ == "__main__":
    df = pd.read_csv("./dataset/SST2/split_train.csv", '\t')
    dataset = SentimentDataset(df)
