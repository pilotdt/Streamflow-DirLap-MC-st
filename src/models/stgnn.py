import torch
import torch.nn as nn
from .layers import *
from training.utils import build_advection_operator


class STGCNGraphConv(nn.Module):
    # STGCNGraphConv contains 'TGTND TGTND TNFF' structure
    # GraphConv is the graph convolution from GCN.
    # GraphConv is not the first-order ChebConv, because the renormalization trick is adopted.
    # Be careful about over-smoothing.
        
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (GraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (GraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normalization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(self, config, A, gso, blocks, n_vertex, add_storage=False):
        super(STGCNGraphConv, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(STConvBlock(config["Kt"], config["Ks"], n_vertex, blocks[l][-1], blocks[l+1], config["act_func"], config["graph_conv_type"], gso, config["enable_bias"], config["droprate"]))
        self.st_blocks = nn.Sequential(*modules)
        Ko = config["n_his"] - (len(blocks) - 3) * 2 * (config["Kt"] - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, config["act_func"], config["enable_bias"], config["droprate"])
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=config["enable_bias"])
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=config["enable_bias"])
            self.relu = nn.ReLU()
            self.do = nn.Dropout(p=config["droprate"])
        self.A=A
        self.add_storage = add_storage
        if self.add_storage:
            self.learn_stor = nn.Parameter(torch.ones(n_vertex) * 0.0001) 
        else:
            self.learn_stor = None
    def forward(self, x):
        if self.add_storage:
            L_dir = build_advection_operator(A=self.A, add_storage=self.add_storage, learn_stor=self.learn_stor)
        x = x.permute(0, 3, 1, 2)
        x = self.st_blocks(x)
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)
        x = x.permute(0, 1, 3, 2)
        if self.add_storage:
            return x, L_dir
        else:
            return x