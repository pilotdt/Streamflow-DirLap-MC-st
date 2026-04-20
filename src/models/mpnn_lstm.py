import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch.nn.utils.rnn import pack_padded_sequence
from training.utils import build_advection_operator
from .base_model import BaseModel 


class MPNN_LSTM(BaseModel):
    def __init__(self, nfeat: int , 
                 n_hid: int , 
                 nout: int , 
                 n_nodes: int, 
                 window: int, 
                 dropout: float,
                 adj,
                 add_storage: bool,
                 horizon: int,
                 seq_len=None,
                 use_packing=False,
                 ):
        """
        Parameters:
        nfeat (int): Number of features
        n_hid (int): Hidden size
        nout (int): Number of output features
        n_nodes (int): Number of nodes
        window (int): Window size
        dropout (float): Dropout rate
        
        Returns:
        x (torch.Tensor): Output of the model
        """
        super(MPNN_LSTM, self).__init__()
        self.window = window
        self.n_nodes = n_nodes
        self.n_hid = n_hid
        self.nfeat = nfeat
        self.horizon = horizon
        self.adj = adj
        self.conv1 = GCNConv(nfeat, n_hid)
        self.conv2 = GCNConv(n_hid, n_hid)
        self.seq_len = seq_len
        self.use_packing = use_packing
        self.add_storage = add_storage

        if self.add_storage:
            self.learn_stor = nn.Parameter(torch.ones(self.n_nodes) * 0.0001) 
        else:
            self.learn_stor = None
        
        self.bn1 = nn.BatchNorm1d(n_hid)
        self.bn2 = nn.BatchNorm1d(n_hid)

        self.fc2 = nn.Linear(n_nodes*n_hid, n_nodes)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()


        self.encoder = nn.LSTM(
            input_size=n_hid*n_nodes,
            hidden_size=n_hid,
            num_layers=1,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=n_nodes,    
            hidden_size=n_hid,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(n_hid, n_nodes)

    def encode(self, x):
        if self.use_packing and self.seq_len is not None:
            packed = pack_padded_sequence(
                x, lengths=self.seq_len, batch_first=True, enforce_sorted=False
            )
            _, (h, c) = self.encoder(packed)
        else:
            _, (h, c) = self.encoder(x)

        return h, c

    def decode(self, x, h, c, B, N, output_dim):
        # y_prev = torch.zeros(B, output_dim, device=h.device)
        B, T, O = x.shape
        x = x.reshape(B, T, N, int(O/N))

        y_prev = x[:, -1, :, 0] 
        outputs = []

        for _ in range(self.horizon):
            y_prev_seq = y_prev.unsqueeze(1) 
            out, (h, c) = self.decoder(y_prev_seq, (h, c))
            y = self.fc(out[:, -1])
            outputs.append(y)
            y_prev = y

        return torch.stack(outputs, dim=1)
        
        
    def forward(self, x):
        lst = list()
        if self.add_storage:
            L_dir = build_advection_operator(A=self.adj, add_storage=self.add_storage, learn_stor=self.learn_stor)


        weight = self.adj.coalesce().values()
        adj = self.adj.coalesce().indices()
        input = x
        skip = x.view(-1,self.window,self.n_nodes,self.nfeat)
       
        skip = torch.transpose(skip, 1, 2).reshape(-1,self.window,self.nfeat)
        x = self.relu(self.conv1(x, adj, edge_weight=weight))
        # print(x.shape)
        # x = x.permute(0, 2, 3, 1)
        # print(x.shape)
        # x = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])
        # print(x.shape)
        # x = self.bn1(x)
        # print(x.shape)
        # x = x.reshape(x.shape[0]//self.n_nodes,self.n_nodes,self.n_hid,self.window)
        # print(x.shape)
        # x = x.permute(0, 3, 1, 2)
        # print(x.shape)
        # x = self.dropout(x)
        # print(x.shape)
        lst.append(x)
        
        # x = self.relu(self.conv2(x, adj, edge_weight=weight))
        # x = x.permute(0, 2, 3, 1)
        # x = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])
        # x = self.bn2(x)
        # x = x.reshape(x.shape[0]//self.n_nodes,self.n_nodes,self.n_hid,self.window)
        # x = x.permute(0, 3, 1, 2)
        # x = self.dropout(x)
        # lst.append(x)
        
        x = torch.cat(lst, dim=3)
        # print(x.shape)
        
        # reshape to (seq_len, batch_size , hidden) to fit the lstm
        x = x.view(-1, self.window, self.n_nodes, x.size(3))
        # print(x.shape)
        B, T, N, F = x.shape
        # print(x.shape)
        # print(a)
        
        x_joint = x.reshape(B, T, N * F)
        # x_joint = self.fc2(x_joint)
        h, c = self.encode(x_joint)


        preds = self.decode(x_joint, h, c, B, N, output_dim=self.fc.out_features)
        preds = preds.unsqueeze(axis=-1)
        if self.add_storage:
            return preds, L_dir
        else:
            return preds



