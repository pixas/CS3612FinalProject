import argparse
from typing import List

import numpy as np
import torch

from model.model_sst2 import SST2Model

from torch import Tensor
from tqdm import tqdm

from .utils import load_sst2_data, visualize_pca, visualize_tsne

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument('ckpt', type=str)
    parser.add_argument('--pca', action='store_true')
    parser.add_argument('--tsne', action='store_true')

    parser.add_argument("--hidden_dims", default=[512, 512], type=List[int])
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    args = parser.parse_args()
    return args


def main():
    args = make_argparser()
    eval_pca = args.pca
    eval_tsne = args.tsne
    
    ckpt_dir = args.ckpt
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = SST2Model(args.hidden_dims, dropout=args.dropout).to(device)
    model_parameter = torch.load(ckpt_dir)
    model.load_state_dict(model_parameter['model'])
    _, _, test_loader = load_sst2_data(args.data, args.batch_size, args.num_workers)
    
    model.eval()
    top1_acc = 0.0
    total = 0.0
    valid_batch_data: List[Tensor]
    predicted_result: np.ndarray
    with torch.no_grad():
        length = len(test_loader)
        with tqdm(total=length) as t:
            for data in test_loader:
                t.set_description("Evaluating")
                sentences, attention_mask, labels, nonzero_index = data
                # run the model on the test set to predict labels
                sentences, labels = sentences.to(device), labels.to(device)
                attention_mask = attention_mask.to(device)
                nonzero_index = nonzero_index.to(device)
                outputs, inter_feature = model(sentences, attention_mask, nonzero_index, True)
                # the label with the highest energy will be our prediction
                _, predicted = torch.max(outputs.data, 1)
                if True:
                    valid_batch_data = [i.detach().cpu().numpy().reshape(labels.shape[0], -1) for i in inter_feature]
                    predicted_result = predicted.detach().cpu().numpy()
                    
                total += labels.size(0)
                top1_acc += (predicted == labels).sum().item()
                t.set_postfix({'top1': top1_acc / total})
                t.update(1)
    top1_acc = (100 * top1_acc / total)
    print("Accuracy: {:.2f}%".format(top1_acc))

    if eval_pca:
        visualize_pca(valid_batch_data, label=predicted_result, task='sst2')
        
    if eval_tsne:
        visualize_tsne(valid_batch_data, label=predicted_result, task='sst2')



if __name__ == "__main__":
    main()
