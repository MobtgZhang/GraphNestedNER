import torch
from torch import Tensor

from torch_geometric.nn import MessagePassing

class CapsuleGATNet(MessagePassing):
    def __init__(self):
        super(CapsuleGATNet,self).__init__()
    def forward(self,):
        pass
    def message(self, x_j: Tensor):
        pass
    def update(self, inputs: Tensor):
        pass
    def __str__(self,):
        pass
    def __repr__(self):
        pass
