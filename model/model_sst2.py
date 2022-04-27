from typing import List
import torch 
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from scripts.dataset import model_class, tokenizer_class, pretrained_weights
rnn = nn.LSTM(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(input, (h0, c0))



class LSTMLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super(LSTMLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim 
        
        self.proj_ii = nn.Linear(input_dim, output_dim, True)
        self.proj_hi = nn.Linear(hidden_dim, output_dim, True)
        
        self.proj_if = nn.Linear(input_dim, output_dim, True)
        self.proj_hf = nn.Linear(hidden_dim, output_dim, True)
        
        self.proj_ig = nn.Linear(input_dim, output_dim, True)
        self.proj_hg = nn.Linear(hidden_dim, output_dim, True)
        
        self.proj_io = nn.Linear(input_dim, output_dim, True)
        self.proj_ho = nn.Linear(hidden_dim, output_dim, True)
        
    
    def forward(self, input: Tensor, 
                h_0: Tensor = None, 
                c_0: Tensor = None, 
                attention_mask: Tensor=None,
                nonzero_index: Tensor=None):
        B, T, input_dim = input.shape
        if h_0 is None:
            h_0 = torch.zeros((B, 1, self.hidden_dim)).to(input.device)
        if c_0 is None:
            c_0 = torch.zeros((B, 1, self.hidden_dim)).to(input.device)
        
        B, _, hidden_dim = h_0.shape
        assert hidden_dim == self.hidden_dim, "Dimension of hidden state does not match initialization"
        B, _, hidden_dim = c_0.shape
        assert hidden_dim == self.hidden_dim, "Dimension of cell state does not match initialization"
        
        ht_list = [h_0]
        ct_list = [c_0]
        
        for i in range(T):
            xt = input[:, i, :]
            ht_1 = ht_list[-1][:, 0, :]
            ct_1 = ct_list[-1][:, 0, :]
            ht, ct = self.forward_one_cell(xt, ht_1, ct_1)
            ht_list.append(ht[:, None, :])
            ct_list.append(ct[:, None, :])
        
        hts = torch.cat(ht_list[1:], 1)
        cts = torch.cat(ct_list[1:], 1)
        
        cls_token = hts.gather(1, nonzero_index.unsqueeze(-1).repeat(1, 1, self.output_dim)).squeeze(1)

        hts = hts * (attention_mask.unsqueeze(-1))
        cts = cts * (attention_mask.unsqueeze(-1))
        
        return hts, hts, cts, cls_token
        
    def forward_one_cell(self, xt: Tensor, ht_1: Tensor, ct_1: Tensor):
        it = F.sigmoid(self.proj_ii(xt) + self.proj_hi(ht_1))
        ft = F.sigmoid(self.proj_if(xt) + self.proj_hf(ht_1))
        gt = F.tanh(self.proj_ig(xt) + self.proj_hg(ht_1))
        ot = F.sigmoid(self.proj_io(xt) + self.proj_ho(ht_1))
        ct = ft * ct_1 + it * gt
        ht = ot * F.tanh(ct)
        return ht, ct
        

class SST2Model(nn.Module):
    def __init__(self, hidden_dims: List[int], output_class: int = 2,
                 dropout: float = 0.1) -> None:
        super(SST2Model, self).__init__()
        self.embed_dim = 768
        self.hidden_dims = hidden_dims
        self.embed = model_class.from_pretrained(pretrained_weights)
        for layer in list(self.embed.parameters()):
            layer.requires_grad = False
        
        self.lstm1 = LSTMLayer(self.embed_dim, self.hidden_dims[0], self.hidden_dims[0])
        self.lstm2 = LSTMLayer(self.hidden_dims[0], self.hidden_dims[0], self.hidden_dims[1])
        self.dropout = dropout
        self.fc1 = nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] * 4)
        self.fc2 = nn.Linear(self.hidden_dims[-1] * 4, output_class)
        
    def forward(self, x, attention_mask, nonzero_index, return_inter=False):
        return_features = []
        embed = self.embed(**{'input_ids': x, 'attention_mask':attention_mask})[0]
        return_features.append(embed[:, 0, :])
        
        x, hs, cs, _ = self.lstm1(embed, attention_mask=attention_mask, nonzero_index=nonzero_index)
        x = F.dropout(x, self.dropout, self.training)
        return_features.append(x[:, 0, :])
        
        x, hs, cs, cls_token = self.lstm2(x, attention_mask=attention_mask, nonzero_index=nonzero_index)
        x = F.dropout(x, self.dropout, self.training)
        return_features.append(x[:, 0, :])
        
        x = self.fc1(cls_token)
        x = F.relu(x)
        return_features.append(x)
        x = F.log_softmax(self.fc2(x), -1)
        if return_inter:
            return x, return_features
        return x
        
        
        
if __name__ == "__main__":
    sentences = ["my best friend is Lily.", "I totally own the world"]
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    
    input_ids = tokenizer(sentences, add_special_tokens=True, padding=True, return_tensors='pt')
    print(input_ids)
    print(model(**input_ids))