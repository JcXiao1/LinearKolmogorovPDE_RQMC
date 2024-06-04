import torch
from torch import nn


EPSILON = 1e-08

def swish(x):
    return x* torch.sigmoid(x)

class MLPNet(torch.nn.Module):
    def __init__(self,dim,width,depth):
        super(MLPNet, self).__init__()
        self.linear_layers = nn.ModuleList([nn.Linear(dim, width, bias=True)])
        self.linear_layers += [
            nn.Linear(width, width, bias=True) for _ in range(depth-1)
        ]
        self.linear_layers.append(nn.Linear(width, 1, bias=True))
        self.norm_layers = nn.ModuleList(
            [nn.BatchNorm1d(width, eps= EPSILON) for _ in range(depth)]
        )
        self.act = swish

    def forward(self, x):
        y = self.linear_layers[0](x)
        for i, linear in enumerate(self.linear_layers[1:]):
            y = self.norm_layers[i](y)
            y = self.act(y)
            y = linear(y)
        return y


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)  
        
