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
    
# class VAE(nn.Module):

#     def __init__(self, input_dim=784, h_dim=400, z_dim=20):
#         # 调用父类方法初始化模块的state
#         super(VAE, self).__init__()

#         self.input_dim = input_dim
#         self.h_dim = h_dim
#         self.z_dim = z_dim

#         # 编码器 ： [b, input_dim] => [b, z_dim]
#         self.fc1 = nn.Linear(input_dim, h_dim)  # 第一个全连接层
#         self.fc2 = nn.Linear(h_dim, z_dim)  # mu
#         self.fc3 = nn.Linear(h_dim, z_dim)  # log_var

#         # 解码器 ： [b, z_dim] => [b, input_dim]
#         self.fc4 = nn.Linear(z_dim, h_dim)
#         self.fc5 = nn.Linear(h_dim, input_dim)

#     def forward(self, x):
#         """
#         向前传播部分, 在model_name(inputs)时自动调用
#         :param x: the input of our training model [b, batch_size, 1, 28, 28]
#         :return: the result of our training model
#         """
#         batch_size = x.shape[0]  # 每一批含有的样本的个数
#         # flatten  [b, batch_size, 1, 28, 28] => [b, batch_size, 784]
#         # tensor.view()方法可以调整tensor的形状，但必须保证调整前后元素总数一致。view不会修改自身的数据，
#         # 返回的新tensor与原tensor共享内存，即更改一个，另一个也随之改变。
#         x = x.view(batch_size, self.input_dim)  # 一行代表一个样本

#         # encoder
#         mu, log_var = self.encode(x)
#         # reparameterization trick
#         sampled_z = self.reparameterization(mu, log_var)
#         # decoder
#         x_hat = self.decode(sampled_z)
#         # reshape
#         return mu, log_var, x_hat

#     def encode(self, x):
#         """
#         encoding part
#         :param x: input image
#         :return: mu and log_var
#         """
#         h = F.relu(self.fc1(x))
#         mu = self.fc2(h)
#         log_var = self.fc3(h)

#         return mu, log_var

#     def reparameterization(self, mu, log_var):
#         """
#         Given a standard gaussian distribution epsilon ~ N(0,1),
#         we can sample the random variable z as per z = mu + sigma * epsilon
#         :param mu:
#         :param log_var:
#         :return: sampled z
#         """
#         sigma = torch.exp(log_var * 0.5)
#         eps = torch.rand_like(sigma)
#         return mu + sigma * eps  # 这里的“*”是点乘的意思

#     def decode(self, z):
#         """
#         Given a sampled z, decode it back to image
#         :param z:
#         :return:
#         """
#         h = F.relu(self.fc4(z))
#         x_hat = torch.sigmoid(self.fc5(h))  # 图片数值取值为[0,1]，不宜用ReLU
#         x_hat = x_hat.view(x_hat.shape[0], 1, 28, 28)
#         return x_hat
# class VAE(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(784, 500)
#         self.fc21 = nn.Linear(500, 2)  # fc21 for mean of Z
#         self.fc22 = nn.Linear(500, 2)  # fc22 for log variance of Z
#         self.fc3 = nn.Linear(2, 500)
#         self.fc4 = nn.Linear(500, 784)

#     def encode(self, x):
#         x = x.reshape(x.shape[0], -1)
#         h1 = F.relu(self.fc1(x))
#         mu = self.fc21(h1)
#         # I guess the reason for using logvar instead of std or var is that
#         # the output of fc22 can be negative value (std and var should be positive)
#         logvar = self.fc22(h1)
#         return mu, logvar

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.rand_like(std)
#         return mu + eps*std

#     def decode(self, z):
#         h3 = F.relu(self.fc3(z))
#         return torch.sigmoid(self.fc4(h3)).reshape(z.shape[0], 1, 28, 28)

#     def forward(self, x):
#         # x: [batch size, 1, 28,28] -> x: [batch size, 784]
#         x = x.view(-1, 784)
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         return mu, logvar, self.decode(z)
        
if __name__ == "__main__":
    A = nn.ConvTranspose2d(128, 64, [3, 3], [2, 2], [1, 1], [1, 1])
    x = torch.randn((2, 128, 7, 7))
    print(A(x).shape)