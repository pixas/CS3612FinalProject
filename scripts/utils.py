import os
from typing import List

import numpy as np
import seaborn as sns
import pandas as pd 
import torch
from matplotlib import pyplot as plt
from model.model_mnist import MnistModel
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .dataset import SentimentDataset
sns.set(rc={'figure.figsize':(11.7,8.27)})

def plot_acc_loss(loss_list, acc_list, data_mode='train', task='mnist'):
    length = len(loss_list)
    f, (ax1, ax2) = plt.subplots(1, 2)
    epoch_list = np.arange(length)
    ax1.plot(epoch_list, loss_list)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel("Loss")
    
    ax2.plot(epoch_list, acc_list)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel("Accuracy")
    
    plt.savefig("./images/{}-{}-loss-acc.png".format(task, data_mode), dpi=600)
    plt.cla()
    


def load_mnist_data(data_dir, bsz, num_workers):
    if not os.path.exists(data_dir):
        download=True
    else:
        download=False
    data_train = datasets.MNIST(root = data_dir,
                                transform=transforms.ToTensor(),
                                train = True,
                                download = download)

    data_test = datasets.MNIST(root=data_dir,
                            transform=transforms.ToTensor(),
                            train=False,
                            download=download)

    train_loader = DataLoader(data_train, bsz, True, num_workers=num_workers)
    test_loader = DataLoader(data_test, bsz, False, num_workers=num_workers)
    
    return train_loader, test_loader

def load_sst2_data(data_dir, bsz, num_workers):
    train_data = pd.read_csv(os.path.join(data_dir, 'split_train.csv'), '\t')
    test_data = pd.read_csv(os.path.join(data_dir, 'split_test.csv'), '\t')
    dev_data = pd.read_csv(os.path.join(data_dir, 'split_dev.csv'), '\t')
    data_train = SentimentDataset(train_data)
    data_test = SentimentDataset(test_data)
    data_dev = SentimentDataset(dev_data)
    
    train_loader = DataLoader(data_train, bsz, True, num_workers=num_workers)
    test_loader = DataLoader(data_test, bsz, False, num_workers=num_workers)
    dev_loader = DataLoader(data_dev, bsz, False, num_workers=num_workers)
    
    return train_loader, test_loader, dev_loader

def load_fashion_mnist_data(data_dir, bsz, num_workers):
    if not os.path.exists(data_dir):
        download=True
    else:
        download=False
    
    train_set = datasets.FashionMNIST(
        root=data_dir,
        transform=transforms.ToTensor(),
        train=True,
        download=download
    )
    
    test_set = datasets.FashionMNIST(
        root=data_dir,
        transform=transforms.ToTensor(),
        train=False,
        download=download
    )
    
    train_loader = DataLoader(train_set, bsz, True, num_workers=num_workers)
    test_loader = DataLoader(test_set, bsz, False, num_workers=num_workers)
    
    return train_loader, test_loader
    
    


def save_model(model: MnistModel, save_dir: str, top1acc=None, top5acc=None, epoch=None, description=None, loss=None):
    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'top1acc': top1acc,
        'top5acc': top5acc,
        'best_loss': loss
    }, save_dir if description is None else save_dir + description)




def visualize_pca(batch_data: List[np.ndarray], label: np.ndarray, n: int = 2, task: str='mnist'):
    length = len(batch_data)
    convert = lambda x: 'C{}'.format(x)
    color = [convert(x) for x in label]
    pca_list = [MyPCA(n) for i in range(length)]

    pca_list = [pca_list[i].fit(batch_data[i]) for i in range(length)]
    
    data_decomposed = [pca_list[i].transform(batch_data[i]) for i in range(length)]
    palette = sns.color_palette("bright", 10 if task == 'mnist' else 2)
    for i in range(length):
        data = data_decomposed[i]
        # plt.scatter(data[:, 0], data[:, 1], c=color)
        fig = sns.scatterplot(data[:,0], data[:,1], hue=label, legend='full', palette=palette)
        scatter_fig = fig.get_figure()
        scatter_fig.savefig("./images/{}_feature_map{}_pca.png".format(task, i + 1), dpi=600)
        plt.cla()

def visualize_tsne(batch_data: List[np.ndarray], label: np.ndarray, n: int = 2, task: str='mnist'):
    length = len(batch_data)
    convert = lambda x: 'C{}'.format(x)
    color = [convert(x) for x in label]
    tsne_list = [TSNE(n, learning_rate='auto', init='pca') for i in range(length)]
    # tsne_list = [tsne_list[i].fit(batch_data[i]) for i in range(length)]
    data_decomposed = [tsne_list[i].fit_transform(batch_data[i]) for i in range(length)]
    palette = sns.color_palette("bright", 10 if task == 'mnist' else 2)
    for i in range(length):
        data = data_decomposed[i]
        # plt.scatter(data[:, 0], data[:, 1], c=color)
        fig = sns.scatterplot(data[:,0], data[:,1], hue=label, legend='full', palette=palette)
        scatter_fig = fig.get_figure()
        scatter_fig.savefig("./images/{}_feature_map{}_tsne.png".format(task, i + 1), dpi = 600)
        plt.cla()
        # plt.savefig("./images/{}_feature_map{}_tsne.png".format(task, i + 1), dpi=600)
        # plt.show()



class MyPCA:
    def __init__(self, n_components: int = 2) -> None:
        self.n = n_components
    
    def fit(self, X: np.ndarray):
        n_samples, n_features = X.shape
        assert n_features >= self.n, "Main components must be less than or equal to total feature numbers"
        mean_val = X.mean(0, keepdims=True)
        X = X - mean_val
        U, sigma, VT = np.linalg.svd(X, False)
        self.P = U[:, :self.n]
        max_abs_cols = np.argmax(np.abs(U), axis=0)
        signs = np.sign(U[max_abs_cols, range(U.shape[1])])
        U *= signs
        VT *= signs[:, np.newaxis]
        self.P = VT[:self.n]
        # scatter_matrix = norm_x.T @ norm_x
        # eig_val, eig_vec = np.linalg.eig(scatter_matrix)

        # eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]
        # eig_pairs.sort(reverse=True, key=lambda x: x[0])
        # # select the top k eig_vec
        # feature=np.array([ele[1] for ele in eig_pairs[:self.n]])
        # self.P = feature.T
        return self
    
    def transform(self, X):
        return X @ self.P.T


if __name__ == "__main__":
    x, y = load_fashion_mnist_data("./dataset/", 4, 0)
    
        
    