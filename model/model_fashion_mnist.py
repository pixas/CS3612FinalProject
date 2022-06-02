import torch 
import torch.nn as nn 
from typing import List
from torch import Tensor
import torch.nn.functional as F
import numpy as np 


class Encoder(nn.Module):
    HEIGHT=28
    WIDTH=28
    def __init__(self, input_dim: int, 
                 output_dim: int, 
                 hidden_dims: List[int], 
                 dropout: float=0.1,
                 strides: List[List[int]] = [[2, 2]],
                 conv_kernels: List[List[int]] = [[5, 5], [5, 5]]) -> None:
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim 
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        self.num_layers = len(strides)

        assert self.num_layers == len(conv_kernels)
        self.convs = nn.ModuleList([
            nn.Conv2d(input_dim if i == 0 else hidden_dims[i - 1], 
                      hidden_dims[i], conv_kernels[i], 
                      [1, 1], 
                      [conv_kernels[i][0] // 2, conv_kernels[i][1] // 2])
        for i in range(self.num_layers)])
        self.maxpool_layers = nn.ModuleList([
            nn.MaxPool2d(conv_kernels[i], strides[i], [conv_kernels[i][0] // 2, conv_kernels[i][1] // 2])
        for i in range(self.num_layers)])
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(hidden_dims[i])
        for i in range(self.num_layers - 1)])
        self.strides = np.array(strides)
        self.output_size = [self.HEIGHT // (np.prod(self.strides[:, 0])), self.WIDTH // (np.prod(self.strides[:, 1]))]
        self.fc_mu = nn.Linear(output_dim * self.output_size[0] * self.output_size[1], output_dim)
        self.fc_logvar = nn.Linear(output_dim * self.output_size[0] * self.output_size[1], output_dim)

    def forward(self, x: Tensor):
        b, f, h, w = x.shape 

        for i in range(self.num_layers - 1):

            x = self.convs[i](x)

            x = self.bn_layers[i](x)
            x = F.relu(x)

            x = self.maxpool_layers[i](x)
            x = F.dropout(x, self.dropout, self.training)

        x = self.convs[-1](x)
        x = self.maxpool_layers[-1](x)
        assert x.shape[2] == self.output_size[0] and x.shape[-1] == self.output_size[1]
        x = x.reshape(b, -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
            
    
class Decoder(nn.Module):
    HEIGHT=28
    WIDTH=28
    def __init__(self, input_dim: int, 
                 output_dim: int, 
                 hidden_dims: List[int], 
                 dropout: float=0.1,
                 strides: List[List[int]] = [[2, 2]],
                 conv_kernels: List[List[int]] = [[5, 5], [5, 5]]) -> None:
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim 
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        self.num_layers = len(strides)
        assert self.num_layers == len(conv_kernels)
        self.strides = np.array(strides)
        self.output_size = [self.HEIGHT // (np.prod(self.strides[:, 0])), self.WIDTH // (np.prod(self.strides[:, 1]))]
        self.fc1 = nn.Linear(input_dim, input_dim * self.output_size[0] * self.output_size[1])

        
        
        self.convs = nn.ModuleList([
            nn.ConvTranspose2d(input_dim if i == 0 else hidden_dims[i], 
                      hidden_dims[i+1] if i != self.num_layers - 1 else output_dim, conv_kernels[i], 
                      strides[i], 
                      [conv_kernels[i][0] // 2, conv_kernels[i][1] // 2],
                      [conv_kernels[i][0] // 2, conv_kernels[i][1] // 2])
        for i in range(self.num_layers)])
        
        # self.maxpool_layers = nn.ModuleList([
        #     nn.MaxPool2d(conv_kernels[i], strides[i], [conv_kernels[i][0] // 2, conv_kernels[i][1] // 2])
        # for i in range(self.num_layers)])
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(hidden_dims[i+1])
        for i in range(self.num_layers - 1)])
    
    def forward(self, z: Tensor):
        x = self.fc1(z)
        x = x.reshape(x.shape[0], -1, self.output_size[0], self.output_size[1])

        for i in range(self.num_layers - 1):
            x = self.convs[i](x)
            x = self.bn_layers[i](x)
            x = F.relu(x)

        x = self.convs[-1](x)
        x = torch.sigmoid(x)
        assert x.shape[2] == self.HEIGHT and x.shape[3] == self.WIDTH
        return x 


class VAE(nn.Module):
    HEIGHT = 28
    WIDTH = 28
    def __init__(self,
                 encoded_dim: int,
                 hidden_dims: int,
                 dropout: float = 0.1) -> None:
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(self.WIDTH * self.HEIGHT, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc_mu = nn.Linear(hidden_dims[1], encoded_dim)
        self.fc_logvar = nn.Linear(hidden_dims[1], encoded_dim)
        
        self.fc_3 = nn.Linear(encoded_dim, hidden_dims[1])
        self.fc_4 = nn.Linear(hidden_dims[1], hidden_dims[0])
        self.fc_output = nn.Linear(hidden_dims[0], self.WIDTH * self.HEIGHT)
        
        self.dropout = dropout 

    def encode(self, x: Tensor):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.fc2(x))
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        return mu, logvar
    
    def decode(self, z: Tensor):
        x = F.relu(self.fc_3(z))
        x = F.relu(self.fc_4(x))
        output = self.fc_output(x)
        return torch.sigmoid(output).reshape(output.shape[0], 1, self.HEIGHT, self.WIDTH)
    
    def forward(self, x: Tensor):
        x = x.reshape(x.shape[0], -1)
        mu, logvar = self.encode(x)

        z = self._reparameterize(mu, logvar)
        x_prime = self.decode(z)
        return mu, logvar, x_prime
    
    def _reparameterize(self, mu: Tensor, logvar: Tensor):
        # b x f
        eps = torch.randn_like(mu, device=mu.device)
        return mu + eps * torch.exp(logvar / 2)
    

if __name__ == "__main__":
    A = nn.ConvTranspose2d(128, 64, [3, 3], [2, 2], [1, 1], [1, 1])
    x = torch.randn((2, 128, 7, 7))
    print(A(x).shape)