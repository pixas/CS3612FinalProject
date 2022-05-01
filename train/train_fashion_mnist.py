from argparse import Namespace
import os
from typing import List
from matplotlib import pyplot as plt
import numpy as np
import torch
import argparse
import torch.nn.functional as F
from model.model_fashion_mnist import VAE
from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader

from scripts.utils import load_fashion_mnist_data, save_model
from tqdm import tqdm
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enabled = False
def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--num_layers", default=2, type=int)

    parser.add_argument("--hidden_dims", default=[512, 64], type=List[int])
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--ckpt_dir", default=None, type=str)
    parser.add_argument("--encoded_dim", default=32, type=int)
    args = parser.parse_args()
    return args

def KL_divergence(mu: Tensor, logvar: Tensor):

    return -0.5 * torch.sum(logvar + 1 - mu ** 2 - logvar.exp())

def train_one_epoch(model: VAE, dataloader: DataLoader, loss_fn, optimizer: torch.optim.Adam, epoch: int, device):
    
    model.train()
    running_loss = 0
    length = len(dataloader)

    total_seen = 0
    with tqdm(total=length) as t:
        for i, (images, labels) in enumerate(dataloader, 0):
            t.set_description('Epoch {:03d}'.format(epoch))
            # get the inputs
            images = images.to(device)
            

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            mu, logvar, outputs = model(images)

            # compute the loss based on model output and real labels
            bce_loss = F.binary_cross_entropy(outputs, images, reduction='sum')
            # plt.imshow(images[0, 0].detach().cpu().numpy(), 'gray')
            # plt.show()
            # print(images[0, 0])
            # plt.imshow(outputs[0, 0].detach().cpu().numpy(), 'gray')
            # plt.show()
            # print(outputs[0, 0])
            # exit()
            kl_loss = KL_divergence(mu, logvar)
            loss = bce_loss + kl_loss
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            total_seen += images.shape[0]
            # Let's print statistics for every 1,000 images
            running_loss += loss.item()     # extract the loss value
            t.set_postfix({'loss': loss.item(), 'KL divergence': kl_loss.item(), 'BCE loss': bce_loss.item()})
            # if i % 1000 == 999:    
            #     # print every 1000 (twice per epoch) 
            #     print('[%d, %5d] loss: %.3f' %
            #             (epoch + 1, i + 1, running_loss / 1000))
            #     # zero the loss
            #     running_loss = 0.0
            t.update(1)
    running_loss /= total_seen

    return running_loss


@torch.no_grad()
def eval_one_epoch(model: VAE, test_loader: DataLoader, device: torch.device):
    model.eval()

    total = 0.0
    running_loss = 0.0

    
    with torch.no_grad():
        length = len(test_loader)
        with tqdm(total=length) as t:
            for data in test_loader:
                t.set_description("Evaluating")
                images, labels = data
                # run the model on the test set to predict labels
                images, labels = images.to(device), labels.to(device)
                mu, logvar, outputs = model(images)
                kl_loss = KL_divergence(mu, logvar)
                mse_loss = F.binary_cross_entropy(outputs, images, reduction='sum')
                loss = kl_loss + mse_loss
                # the label with the highest energy will be our prediction


                total += labels.size(0)
                running_loss += loss.item()
                t.set_postfix({'KL divergence': kl_loss.item(), 'bce loss': mse_loss.item(), 'total loss': loss.item()})
                t.update(1)

    # compute the accuracy over all test images
    loss = running_loss / total
    return loss

def train(args: Namespace):
    num_epochs = args.num_epochs
    lr = args.learning_rate

    hidden_dims = args.hidden_dims
    dropout = args.dropout
    batch_size = args.batch_size
    num_workers = args.num_workers
    data_dir = args.data
    save_dir = args.save_dir
    encoded_dim = args.encoded_dim
    
    
    loss_fn = torch.nn.BCELoss()

    train_loader, test_loader = load_fashion_mnist_data(data_dir, batch_size, num_workers)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model_path = args.ckpt_dir
    model = VAE(encoded_dim, hidden_dims, dropout)
    # model = VAE()
    model.to(device)
    
    if model_path is not None and os.path.exists(model_path):
        state_dict = torch.load(model_path)
        begin_epoch = state_dict['epoch'] + 1

        model.load_state_dict(state_dict['model'])
        best_loss = state_dict['best_loss']
        print("Checkpoint loaded successfully")
    else:
        
        begin_epoch = 0
        
        best_loss = 0x7fffffff

        
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.0001)

    for epoch in range(begin_epoch, begin_epoch + num_epochs):
        loss = train_one_epoch(model, train_loader, loss_fn, optimizer, epoch, device)
        
        eval_loss = eval_one_epoch(model, test_loader, device)
        print("Epoch {:02d}, Training Loss: {:.3f}".format(epoch, loss))
        print("Epoch {:02d}, Testing Loss: {:.3f}".format(epoch, eval_loss))

        if should_early_stop(eval_loss, best_loss):
            break
        if eval_loss < best_loss:
            save_model(model, save_dir, loss=eval_loss, epoch=epoch, description='best.pth')
            best_loss = eval_loss
        save_model(model, save_dir, epoch=epoch, description="last.pth", loss=eval_loss)

def should_early_stop(eval_loss, best_loss):
    # timer = 0
    timer = getattr(should_early_stop, 'timer', 0)

    if eval_loss > best_loss:
        timer += 1
    else:
        timer = 0
    should_early_stop.timer = timer
    if timer == 3:
        return True
    return False


def main():
    args = make_argparser()
    train(args)    

if __name__ == "__main__":
    main()