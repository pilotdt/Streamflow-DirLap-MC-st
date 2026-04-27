import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from training.utils import build_advection_operator
from .base_model import BaseModel


class GRU(BaseModel):
    def __init__(self, input_dim, output_dim, hidden, num_layers,
                 horizon, seq_len=None, use_packing=False, 
                 add_storage=False, lambda_nl_reg=None, A=None):
        super().__init__()

        self.hidden = hidden
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.horizon = horizon
        self.use_packing = use_packing
        self.add_storage = add_storage
        self.lambda_nl_reg = lamda_nl_reg
        self.A = A

        if  self.add_storage:
            self.learn_stor = nn.Parameter(torch.ones(output_dim) * 0.0001) 
        if self.add_storage==False:
            self.learn_stor = None
        if self.lambda_nl_reg is not None:
            self.a = nn.Parameter(torch.ones(num_stations) * 0.0001)
            self.b = nn.Parameter(torch.ones(num_stations) * 0.0001)
        
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True
        )

        self.decoder = nn.GRU(
            input_size=output_dim,   
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden, output_dim)

    def encode(self, x, seq_len=None):
        if self.use_packing and seq_len is not None:
            packed = pack_padded_sequence(
                x, lengths=self.seq_len, batch_first=True, enforce_sorted=False
            )
            _, h = self.encoder(packed)
        else:
            _, h = self.encoder(x)

        return h 

    def decode(self, x, h, B, N, output_dim):

        B, T, O = x.shape
        x = x.reshape(B, T, N, int(O/N))

        y_prev = x[:, -1, :, 0] 
        outputs = []

        for _ in range(self.horizon):
            y_prev_seq = y_prev.unsqueeze(1) 

            out, h = self.decoder(y_prev_seq, h)
            y = self.fc(out[:, -1])
            outputs.append(y)
            y_prev = y

        return torch.stack(outputs, dim=1)

    def forward(self, x, seq_len=None):
        B, T, N, F = x.shape
        
        if self.add_storage:
            L_dir = build_advection_operator(A=self.A, add_storage=self.add_storage, learn_stor=self.learn_stor)

        x_joint = x.reshape(B, T, N * F)  

        if self.use_packing and seq_len is not None:
            h = self.encode(x_joint, seq_len)
        else:
            h = self.encode(x_joint, None)

        preds = self.decode(x_joint, h, B, N, output_dim=self.fc.out_features)
        if self.add_storage:
            return preds, L_dir
        elif self.lambda_nl_reg is not None:
            return preds, a, b
        else:
            return preds
