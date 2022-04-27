from argparse import Namespace
import os
from typing import List
import numpy as np
import torch
import argparse
import torch.nn.functional as F
from model.model_sst2 import SST2Model
from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader

from scripts.utils import load_sst2_data, save_model, plot_acc_loss
from tqdm import tqdm


def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--hidden_dims", default=[512, 512], type=List[int])
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--ckpt_dir", default=None, type=str)
    
    args = parser.parse_args()
    return args


def train_one_epoch(model: SST2Model, dataloader: DataLoader, loss_fn, optimizer: torch.optim.Adam, epoch: int, device):
    
    model.train()
    running_loss = 0
    top1_acc = 0.0
    length = len(dataloader)
    total_seen = 0
    with tqdm(total=length) as t:
        for i, (sentences, attention_mask, labels, nonzero_index) in enumerate(dataloader, 0):
            t.set_description('Epoch {:03d}'.format(epoch))
            # get the inputs
            sentences = Variable(sentences.to(device))
            labels = Variable(labels.to(device))
            attention_mask = attention_mask.to(device)
            nonzero_index = nonzero_index.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(sentences, attention_mask, nonzero_index)
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

            t.update(1)
    running_loss /= length
    top1_acc = top1_acc * 100 / total_seen
    return running_loss, top1_acc


@torch.no_grad()
def test_acc(model: SST2Model, test_loader: DataLoader, loss_fn, device: torch.device):
    model.eval()
    top1_acc = 0.0
    total = 0.0
    running_loss = 0
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
                outputs = model(sentences, attention_mask, nonzero_index)
                loss = loss_fn(outputs, labels)
                running_loss += loss.item()
                # the label with the highest energy will be our prediction
                _, predicted = torch.max(outputs.data, 1)


                total += labels.size(0)
                top1_acc += (predicted == labels).sum().item()
                t.set_postfix({'top1': top1_acc / total})
                t.update(1)

    # compute the accuracy over all test images
    top1_acc = (100 * top1_acc / total)
    running_loss /= total
    return running_loss, top1_acc


def train(args: Namespace):
    num_epochs = args.num_epochs
    lr = args.learning_rate
    hidden_dims = args.hidden_dims
    dropout = args.dropout
    batch_size = args.batch_size
    num_workers = args.num_workers
    data_dir = args.data
    save_dir = args.save_dir
    
    loss_fn = F.nll_loss
    

    train_loader, test_loader, dev_loader = load_sst2_data(data_dir, batch_size, num_workers)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model_path = args.ckpt_dir
    model = SST2Model(hidden_dims, dropout=dropout)
    model.to(device)
    
    if model_path is not None and os.path.exists(model_path):
        state_dict = torch.load(model_path)
        begin_epoch = state_dict['epoch']
        model = model.load_state_dict(state_dict['model'])
        best_top1 = state_dict['top1acc']

    else:
        
        begin_epoch = 0
        
        best_top1 = 0

        
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    train_loss, train_acc = [], []
    dev_loss, dev_acc = [], []
    for epoch in range(begin_epoch, begin_epoch + num_epochs):
        loss, acc = train_one_epoch(model, train_loader, loss_fn, optimizer, epoch, device)
        train_loss.append(loss)
        train_acc.append(acc)

        loss, top1_acc = test_acc(model, dev_loader, loss_fn, device)
        dev_loss.append(loss)
        dev_acc.append(top1_acc)

        print("Epoch {:02d}, Top1-acc: {:.2f}%".format(epoch, top1_acc))
        if top1_acc > best_top1:
            save_model(model, save_dir, top1_acc, None, epoch, 'top1.pth')
            best_top1 = top1_acc
        save_model(model, save_dir, top1_acc, None, epoch, 'last.pth')
    
    plot_acc_loss(train_loss, train_acc, 'train', 'sst2')
    plot_acc_loss(dev_loss, dev_acc, 'dev', 'sst2')
    

def main():
    args = make_argparser()
    train(args)    

if __name__ == "__main__":
    main()