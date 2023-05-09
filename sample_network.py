# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 14:58:51 2023

@author: Ahmad Al Musawi
First GNN tutorial 
URL: 

    https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html
"""
import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
