from argparse import Namespace
import os
from typing import List
import numpy as np
import torch
import argparse
import torch.nn.functional as F
from model.model_mnist import MnistModel
from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader

from scripts.utils import load_mnist_data, plot_acc_loss, save_model
from tqdm import tqdm

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--conv_kernels", default=[[3, 3], [5, 5]], type=List[List[int]])
    parser.add_argument("--strides", default=[[2, 2], [2, 2]], type=List[List[int]])
    parser.add_argument("--hidden_dims", default=[32, 64], type=List[int])
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--ckpt_dir", default=None, type=str)
    
    args = parser.parse_args()
    return args



def train_one_epoch(model: MnistModel, dataloader: DataLoader, loss_fn, optimizer: torch.optim.Adam, epoch: int, device):
    
    model.train()
    running_loss = 0
    length = len(dataloader)
    top1_acc = 0
    total_seen = 0
    with tqdm(total=length) as t:
        for i, (images, labels) in enumerate(dataloader, 0):
            t.set_description('Epoch {:03d}'.format(epoch))
            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            top1_acc += (predicted == labels).sum().item()
            total_seen += predicted.shape[0]
            # Let's print statistics for every 1,000 images
            running_loss += loss.item()     # extract the loss value
            t.set_postfix({'loss': loss.item()})
            # if i % 1000 == 999:    
            #     # print every 1000 (twice per epoch) 
            #     print('[%d, %5d] loss: %.3f' %
            #             (epoch + 1, i + 1, running_loss / 1000))
            #     # zero the loss
            #     running_loss = 0.0
            t.update(1)
    running_loss /= total_seen
    top1_acc = top1_acc * 100 / total_seen
    return running_loss, top1_acc


def train(args: Namespace):
    num_epochs = args.num_epochs
    lr = args.learning_rate
    num_layers = args.num_layers
    conv_kernels = args.conv_kernels
    strides = args.strides
    hidden_dims = args.hidden_dims
    dropout = args.dropout
    batch_size = args.batch_size
    num_workers = args.num_workers
    data_dir = args.data
    save_dir = args.save_dir
    
    
    loss_fn = F.nll_loss
    print(data_dir)
    train_loader, test_loader = load_mnist_data(data_dir, batch_size, num_workers)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model_path = args.ckpt_dir
    model = MnistModel(num_layers, conv_kernels, strides, hidden_dims, dropout=dropout)
    model.to(device)
    
    if model_path is not None and os.path.exists(model_path):
        state_dict = torch.load(model_path)
        begin_epoch = state_dict['epoch']
        model = model.load_state_dict(state_dict['model'])
        best_top1 = state_dict['top1acc']
        best_top5 = state_dict['top5acc']
    else:
        
        begin_epoch = 0
        
        best_top1 = 0
        best_top5 = 0
        
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    train_loss, train_acc = [], []
    for epoch in range(begin_epoch, begin_epoch + num_epochs):
        loss, acc = train_one_epoch(model, train_loader, loss_fn, optimizer, epoch, device)
        top1_acc, top5_acc = test_acc(model, test_loader, device)
        train_loss.append(loss)
        train_acc.append(acc)
        print("Epoch {:02d}, Top1-acc: {:.2f}%".format(epoch, top1_acc))
        if top1_acc > best_top1:
            save_model(model, save_dir, top1_acc, top5_acc, epoch, 'top1.pth')
            best_top1 = top1_acc
        if top5_acc > best_top5:
            save_model(model, save_dir, top1_acc, top5_acc, epoch, 'top5.pth')
            best_top5 = top5_acc
    
    plot_acc_loss(train_loss, train_acc)

@torch.no_grad()
def test_acc(model: MnistModel, test_loader: DataLoader, device: torch.device):
    model.eval()
    top1_acc = 0.0
    total = 0.0
    top5_acc = 0.0
    
    with torch.no_grad():
        length = len(test_loader)
        with tqdm(total=length) as t:
            for data in test_loader:
                t.set_description("Evaluating")
                images, labels = data
                # run the model on the test set to predict labels
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                # the label with the highest energy will be our prediction
                _, predicted = torch.max(outputs.data, 1)
                _, top5_predicted = torch.topk(outputs.data, 5, -1)

                total += labels.size(0)
                top5_acc += (top5_predicted == labels[:, None]).sum().item()
                top1_acc += (predicted == labels).sum().item()
                t.set_postfix({'top1': top1_acc / total, 'top5': top5_acc / total})
                t.update(1)

    # compute the accuracy over all test images
    top1_acc = (100 * top1_acc / total)
    top5_acc = (100 * top5_acc / total)
    return top1_acc, top5_acc

def main():
    args = make_argparser()
    train(args)    

if __name__ == "__main__":
    main()