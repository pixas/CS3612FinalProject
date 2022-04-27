import torch
import torchvision
from torchvision import datasets, transforms
import os

if __name__ == "__main__":
    if not os.path.exists("./dataset/MNIST"):
        download=True
    else:
        download=False

    data_train = datasets.MNIST(root = "./dataset/",
                                transform=transforms.ToTensor(),
                                train = True,
                                download = download)

    data_test = datasets.MNIST(root='./dataset/',
                            transform=transforms.ToTensor(),
                            train=False,
                            download=download)

    image, label = data_train[0]
    print(image.shape)