import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import transformers

model_class, tokenizer_class, pretrained_weights = \
    transformers.DistilBertModel, transformers.DistilBertTokenizer, 'distilbert-base-uncased'

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

class SentimentDataset(Data.Dataset):
    def __init__(self, data_df: pd.DataFrame) -> None:
        super(SentimentDataset, self).__init__()
        
        print(data_df.iloc[:, 1].value_counts())
        # self.labels = F.one_hot(torch.tensor())
        self.labels = torch.tensor(data_df.iloc[:, 1].values)
        self.length = len(self.labels)


        sentences = list(data_df.iloc[:, 0])
        result = tokenizer(sentences, return_tensors='pt', padding=True, add_special_tokens = True)
        input_ids = result['input_ids']
        attention_mask = result['attention_mask']
        nonzero_index = torch.zeros((self.length, 1), dtype=torch.int64)

        for i in range(self.length):
            x = attention_mask[i]
            x_zero = torch.nonzero(x)
            nonzero_index[i, 0] = x_zero[-1]
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
