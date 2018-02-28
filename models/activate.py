import torch
import torch.nn as nn


def srelu(x):
    return (x + torch.sqrt(x*x+1.0)) / 2.0

class SReLU(nn.Module):
    def __init__(self):
        super(SReLU, self).__init__()

    def forward(self, x):
        return (x + torch.sqrt(x*x+1.0)) / 2.0
