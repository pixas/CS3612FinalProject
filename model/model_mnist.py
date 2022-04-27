import torch 
import torch.nn as nn 
from typing import List
from torch import Tensor
import torch.nn.functional as F
import numpy as np 

class ConvLayer(nn.Module):
    def __init__(self, input_channels: int, output_channel: int, conv_size: List[int], stride: List[int],
                 dropout: float = 0.1) -> None:
        super(ConvLayer, self).__init__()
        
        self.dropout = dropout
        
        self.stride = stride
        
        self.conv = nn.Conv2d(input_channels, output_channel,
                              conv_size, [1, 1], [conv_size[0] // 2, conv_size[1] // 2])
        
        self.maxpool = nn.MaxPool2d(conv_size, self.stride, padding=[conv_size[0] // 2, conv_size[1] // 2])
        self.bn = nn.BatchNorm2d(output_channel)
        
    def forward(self, x: Tensor):
        conv_x = self.conv(x)
        bn_x = self.bn(conv_x)
        relu_x = F.relu(bn_x)
        # x = x + relu_x
        x = self.maxpool(relu_x)

        x = F.dropout(x, self.dropout, self.training)
     
        return x


class MnistModel(nn.Module):
    HEIGHT = 28
    WIDTH = 28
    def __init__(self, num_layers: int, 
                 conv_kernels: List[List[int]], 
                 strides: List[List[int]], 
                 hidden_dims: List[int], 
                 input_dim: int = 1, 
                 output_class: int = 10, 
                 dropout: float = 0.1) -> None:
        super(MnistModel, self).__init__()
        self.num_layers = num_layers
        self.conv_kernels = conv_kernels
        self.output_class = output_class
        self.dropout = dropout
        self.strides = np.array(strides)
        self.convs = nn.ModuleList([])
        input_channel = input_dim
        
        for i in range(num_layers):
            output_channel = hidden_dims[i]
            self.convs.append(ConvLayer(input_channel, output_channel, conv_kernels[i], strides[i],
                                        self.dropout))
            input_channel = hidden_dims[i]
        
        self.output_height = np.ceil(self.HEIGHT / (np.prod(self.strides[:, 0]))).astype(int)
        self.output_width = np.ceil(self.WIDTH / (np.prod(self.strides[:, 1]))).astype(int)
        self.out_proj = nn.Linear(hidden_dims[-1] * self.output_width * self.output_height, output_class)
        
    
    
    
    def forward(self, x: Tensor, return_inter: bool = False):
        inter_layer = []
        for i, conv in enumerate(self.convs):
            x = conv(x)
            inter_layer.append(x)
        
        x = torch.permute(x, [0, 2, 3, 1])
        x = x.reshape([x.shape[0], -1])
        x = self.out_proj(x)
        inter_layer.append(x)
        x = torch.log_softmax(x, -1)
        if return_inter:
            return x, inter_layer
        else:
            return x
