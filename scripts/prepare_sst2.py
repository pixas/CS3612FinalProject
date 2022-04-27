import numpy as np
import pandas as pd 



if __name__ == "__main__":
    df = pd.read_csv("./dataset/SST2/train.tsv", delimiter='\t', header=None)
    df_len = len(df)
    print("Total Sentences: {}".format(df_len))
    train_split = 0.8
    dev_split = 0.1

    train_df = df.iloc[:int(df_len * train_split)]
    test_df = df.iloc[int(df_len * train_split):]
    dev_df = test_df.iloc[:int(df_len * dev_split)]
    test_df = test_df.iloc[int(df_len * dev_split):]

    assert len(test_df) + len(dev_df) + len(train_df) == df_len
    train_df.to_csv("./dataset/SST2/split_train.csv", '\t', index=False)
    dev_df.to_csv("./dataset/SST2/split_dev.csv", '\t', index=False)
    test_df.to_csv("./dataset/SST2/split_test.csv", '\t', index=False)