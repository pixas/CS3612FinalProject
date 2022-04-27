import argparse
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from model.model_mnist import MnistModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import Tensor
from tqdm import tqdm

from .utils import load_mnist_data, visualize_pca, visualize_tsne

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument('ckpt', type=str)
    parser.add_argument('--pca', action='store_true')
    parser.add_argument('--tsne', action='store_true')
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--conv_kernels", default=[[3, 3], [5, 5]], type=List[List[int]])
    parser.add_argument("--strides", default=[[2, 2], [2, 2]], type=List[List[int]])
    parser.add_argument("--hidden_dims", default=[32, 64], type=List[int])
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    args = parser.parse_args()
    return args

def is_valid_batch(x: Tensor):
    result = np.array([(x == i).sum().item() for i in range(10)])
    if (result >= 10).all():
        return True
    return False

def main():
    args = make_argparser()
    eval_pca = args.pca
    eval_tsne = args.tsne
    
    ckpt_dir = args.ckpt
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = MnistModel(args.num_layers, args.conv_kernels, args.strides, args.hidden_dims, dropout=args.dropout).to(device)
    model_parameter = torch.load(ckpt_dir)
    model.load_state_dict(model_parameter['model'])
    _, test_loader = load_mnist_data(args.data, args.batch_size, args.num_workers)
    
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
                images, labels = data
                # run the model on the test set to predict labels
                images, labels = images.to(device), labels.to(device)
                outputs, inter_feature = model(images, True)
                # the label with the highest energy will be our prediction
                _, predicted = torch.max(outputs.data, 1)
                if is_valid_batch(predicted):
                    valid_batch_data = [i.detach().cpu().numpy().reshape(args.batch_size, -1) for i in inter_feature]
                    predicted_result = predicted.detach().cpu().numpy()
                total += labels.size(0)
                top1_acc += (predicted == labels).sum().item()
                t.set_postfix({'top1': top1_acc / total})
                t.update(1)
    top1_acc = (100 * top1_acc / total)
    print("Accuracy: {:.2f}%".format(top1_acc))
    
    if eval_pca:
        visualize_pca(valid_batch_data, label=predicted_result, task='mnist')
        
    if eval_tsne:
        visualize_tsne(valid_batch_data, label=predicted_result, task='mnist')



if __name__ == "__main__":
    main()
