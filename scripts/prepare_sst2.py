import pickle
import numpy as np
import pandas as pd 
from tqdm import tqdm


if __name__ == "__main__":
    df = pd.read_csv("./dataset/SST2/train.tsv", delimiter='\t', header=None)
    df = np.array(df)

    words_dictionary = {}
    count_word = 1

    sentence_ids = []
    sentence_labels = []
    sentence_num = len(df)
    with tqdm(total=sentence_num) as t:
        t.set_description("Counting words in SST-2:")
        for i in range(sentence_num):
            sentence = df[i, 0]

            words_list = sentence.split(' ')
            sentence_id = []
            for j in words_list:
                if j not in words_dictionary:
                    words_dictionary[j] = count_word
                    sentence_id.append(count_word)
                    count_word += 1
                else:
                    sentence_id.append(words_dictionary[j])
            sentence_ids.append(sentence_id)
            sentence_labels.append(df[i,1])
            t.update(1)

            
    max_length = max([len(i) for i in sentence_ids])
    print("SST-2 dictionary size: {}\nMax sentence length: {}\nSST-2 number of sentences: {}".format(count_word - 1, max_length, sentence_num))
    padded_sentence_ids = [i + [0] * (max_length - len(i)) for i in sentence_ids]
    padded_sentence_ids = np.array(padded_sentence_ids)
    sentence_labels = np.array(sentence_labels).reshape(-1, 1)
    sentence_array = np.concatenate([sentence_labels, padded_sentence_ids], axis=-1)
    train_split = 0.8
    dev_split = 0.1

    train_sentences = sentence_array[:int(sentence_num * train_split)]
    test_sentences = sentence_array[int(sentence_num * train_split):]
    dev_sentences = test_sentences[:int(sentence_num * dev_split)]
    test_sentences = test_sentences[int(sentence_num * dev_split):]
    
    dataset_info = {
        'dictionary_size': count_word - 1,
        'max_sentence_length': max_length
    }
    with open("./dataset/SST2/dictionary.pkl", 'wb') as f:
        pickle.dump(dataset_info, f)
    
    np.save("./dataset/SST2/split_train.npy", train_sentences)
    np.save("./dataset/SST2/split_dev.npy", dev_sentences)
    np.save("./dataset/SST2/split_test.npy", test_sentences)
