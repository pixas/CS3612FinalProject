import os
from typing import List

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from model.model_mnist import MnistModel
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 10)

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


def save_model(model: MnistModel, save_dir: str, top1acc=None, top5acc=None, epoch=None, description=None):
    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'top1acc': top1acc,
        'top5acc': top5acc
    }, save_dir if description is None else save_dir + description)




def visualize_pca(batch_data: List[np.ndarray], label: np.ndarray, n: int = 2, task: str='mnist'):
    length = len(batch_data)
    convert = lambda x: 'C{}'.format(x)
    color = [convert(x) for x in label]
    pca_list = [MyPCA(n) for i in range(length)]

    pca_list = [pca_list[i].fit(batch_data[i]) for i in range(length)]
    
    data_decomposed = [pca_list[i].transform(batch_data[i]) for i in range(length)]
    
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
