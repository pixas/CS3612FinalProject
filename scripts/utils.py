import os
import pickle
from typing import List

import numpy as np
import seaborn as sns
import pandas as pd
from sklearn import metrics 
import torch
from matplotlib import pyplot as plt
from model.model_mnist import MnistModel
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .dataset import SentimentDataset
import copy
sns.set(rc={'figure.figsize':(18, 8)})

def plot_acc_loss(loss_list, acc_list, data_mode='train', task='mnist'):
    length = len(loss_list)
    f, (ax1, ax2) = plt.subplots(1, 2)
    # f, ax = plt.subplots(1,1)
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

def load_sst2_data(data_dir, bsz, num_workers, train=True, dev=True, test=True):
    assert train or dev or test, "Must specify one split of dataset"
    train_loader = dev_loader = test_loader = None
    if train:
        train_data = np.load(os.path.join(data_dir, 'split_train.npy'))
        data_train = SentimentDataset(train_data)
        train_loader = DataLoader(data_train, bsz, True, num_workers=num_workers)
    if dev:
        dev_data = np.load(os.path.join(data_dir, 'split_dev.npy'))
        data_dev = SentimentDataset(dev_data)
        dev_loader = DataLoader(data_dev, bsz, False, num_workers=num_workers)
    if test:
        test_data = np.load(os.path.join(data_dir, 'split_test.npy'))
        data_test = SentimentDataset(test_data)
        test_loader = DataLoader(data_test, bsz, False, num_workers=num_workers)
    with open(os.path.join(data_dir, 'dictionary.pkl'), 'rb') as f:
        dictionary = pickle.load(f)
    
    
    
    return train_loader, test_loader, dev_loader, dictionary

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
    sns.set(rc={'figure.figsize':(8, 8)})
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
        x_tick = plt.xticks()[0]
        y_tick = plt.yticks()[0]
        ax = plt.gca()
        ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='both')
        plt.xticks(x_tick, fontsize=17)
        plt.yticks(y_tick, fontsize=17)
        plt.legend(fontsize=17)
        scatter_fig = fig.get_figure()
        scatter_fig.savefig("./images/{}_feature_map{}_pca.png".format(task, i + 1), dpi=600)
        plt.cla()

def visualize_tsne(batch_data: List[np.ndarray], label: np.ndarray, n: int = 2, task: str='mnist', neighbors:int=64):
    sns.set(rc={'figure.figsize':(8, 8)})
    length = len(batch_data)
    convert = lambda x: 'C{}'.format(x)
    color = [convert(x) for x in label]
    tsne_list = [TSNE(n, learning_rate='auto', init='pca') for i in range(length)]
    tsne_list = [MytSNE(n, neighbors=neighbors) for i in range(length)]
    # tsne_list = [tsne_list[i].fit(batch_data[i]) for i in range(length)]
    data_decomposed = [tsne_list[i].fit_transform(batch_data[i]) for i in range(length)]
    palette = sns.color_palette("bright", 10 if task == 'mnist' else 2)
    for i in range(length):
        data = data_decomposed[i]
        # plt.scatter(data[:, 0], data[:, 1], c=color)
        fig = sns.scatterplot(data[:,0], data[:,1], hue=label, legend='full', palette=palette)
        ax = plt.gca()
        ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='both')
        x_tick = plt.xticks()[0]
        y_tick = plt.yticks()[0]
        plt.xticks([float(x) for x in x_tick], fontsize=17)
        plt.yticks([float(y) for y in y_tick], fontsize=17)
        plt.legend(fontsize=17)
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

        return self
    
    def transform(self, X):
        return X @ self.P.T

class MytSNE:
    def __init__(self, n_components=2, max_iter=1000, neighbors=30,
                 learning_rate=200, embed_scale=0.1, tol=1e-5) -> None:
        self.n_components=n_components
        self.max_iter = max_iter
        self.neighbors = neighbors
        self.learning_rate = learning_rate
        self.embed_scale = embed_scale
        self.tol = tol
    
    def fit_transform(self, X: np.ndarray):
        n_samples, n_features = X.shape 
        P = self.calc_matrix_P(X)
        Y = np.random.randn(n_samples, self.n_components) * 1e-4
        Q = self.calc_matrix_Q(Y)
        dy = self.calc_grad(P, Q, Y)
        for i in range(self.max_iter):
            if i == 0:
                Y = Y - self.learning_rate * dy
                Y1 = Y
                error1 = self.calc_loss(P, Q)
            elif i == 1:
                Y = Y - self.learning_rate * dy
                Y2 = Y
                error2 = self.calc_loss(P, Q)
            else:
                YY = Y - self.learning_rate * dy + self.embed_scale * (Y2 - Y1)
                QQ = self.calc_matrix_Q(YY)
                error = self.calc_loss(P, QQ)
                if error > error2:
                    self.learning_rate *= 0.7
                    continue
                elif abs(error - error2) > abs(error2 - error1):
                    self.learning_rate *= 1.2
                Y = YY
                error1 = error2
                error2 = error
                Q = QQ 
                dy = self.calc_grad(P, Q, Y)
                Y1 = Y2 
                Y2 = Y 
            if self.calc_loss(P, Q) < self.tol:
                return Y
            # if np.fmod(i+1,10)==0:
            #     print ('%s iterations the error is %s, Learning Rate is %s'%(str(i+1),str(round(self.calc_loss(P,Q),2)),str(round(self.learning_rate,3))))
        return Y

    def calc_matrix_P(self, X: np.ndarray):
        entropy=np.log(self.neighbors)
        n1,n2=X.shape
        D=np.square(metrics.pairwise_distances(X))
        D_sort=np.argsort(D,axis=1)
        P=np.zeros((n1,n1))
        for i in range(n1):
            Di=D[i,D_sort[i,1:]]
            P[i,D_sort[i,1:]]=self.calc_p(Di,entropy=entropy)
        P=(P+np.transpose(P))/(2*n1)
        P=np.maximum(P,1e-100)
        return P
    
    def calc_p(self, D: np.ndarray, entropy: float, iter_times=50):
        beta=1.0
        H=self.calc_entropy(D,beta)
        error=H-entropy
        k=0
        betamin=-np.inf
        betamax=np.inf
        while np.abs(error)>1e-4 and k<=iter_times:
            if error > 0:
                betamin=copy.deepcopy(beta)
                if betamax==np.inf:
                    beta=beta*2
                else:
                    beta=(beta+betamax)/2
            else:
                betamax=copy.deepcopy(beta)
                if betamin==-np.inf:
                    beta=beta/2
                else:
                    beta=(beta+betamin)/2
            H=self.calc_entropy(D,beta)
            error=H-entropy
            k+=1
        P=np.exp(-D*beta)
        P=P/np.sum(P)
        return P

    def calc_entropy(self, D: np.ndarray, beta: float):
        P=np.exp(-D*beta)
        sumP=sum(P)
        sumP=np.maximum(sumP,1e-200)
        H=np.log(sumP) + beta * np.sum(D * P) / sumP
        return H
    
    
    def calc_matrix_Q(self, Y: np.ndarray):
        n1, n2=Y.shape
        D=np.square(metrics.pairwise_distances(Y))

        Q=(1/(1+D))/(np.sum(1/(1+D))-n1)
        Q=Q/(np.sum(Q)-np.sum(Q[range(n1),range(n1)]))
        Q[np.arange(n1),np.arange(n1)]=0
        Q=np.maximum(Q,1e-100)
        return Q
    
    def calc_grad(self, P: np.ndarray, Q: np.ndarray, Y: np.ndarray):
        n1,n2=Y.shape
        grad=np.zeros((n1,n2))
        for i in range(n1):
            E=(1+np.sum((Y[i,:]-Y)**2,axis=1))**(-1)
            F=Y[i,:]-Y
            G=(P[i,:]-Q[i,:])
            E=E.reshape((-1,1))
            G=G.reshape((-1,1))
            G=np.tile(G,(1,n2))
            E=np.tile(E,(1,n2))
            grad[i,:]=np.sum(4*G*E*F,axis=0)
        return grad

    
    def calc_loss(self, P: np.ndarray, Q: np.ndarray):
        kl_loss = np.sum(P * np.log(P / Q))
        return kl_loss
        

if __name__ == "__main__":
    x, y = load_fashion_mnist_data("./dataset/", 4, 0)
    for i, (images, labels) in enumerate(x):
        print(images.shape)
        plt.imshow(images[0, 0].detach().cpu().numpy(), 'gray')
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.savefig("./images/vae_example.png", dpi=600)
        exit()
        
    