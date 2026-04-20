import torch
import torch.nn as nn
import os
from training.utils import build_advection_operator
from .base_model import BaseModel 
os.environ["TORCH_EXTENSIONS_DIR"] = "/app/.torch_extensions"


from slstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

class sLSTM(BaseModel):
    def __init__(
        self,
        num_stations,
        in_features,
        hidden,
        horizon,
        batch_size,
        xlstm_kernel_size=3, 
        xlstm_num_heads=1,
        xlstm_num_blocks=1,
        xlstm_proj_factor=1.0,
        dropout=0.0,
        add_storage=False,
        A=None
    ):
        super().__init__()

        self.num_stations = num_stations
        self.horizon = horizon
        self.hidden = hidden

        self.input_dim = num_stations * in_features
        self.add_storage = add_storage
        self.A = A
        if  self.add_storage:
            self.learn_stor = nn.Parameter(torch.ones(num_stations) * 0.0001)
        if self.add_storage==False:
            self.learn_stor = None
        self.embed = nn.Linear(self.input_dim, hidden)

        cfg = xLSTMBlockStackConfig(
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda" if torch.cuda.is_available() else "vanilla",
                    num_heads=xlstm_num_heads,
                    conv1d_kernel_size=xlstm_kernel_size,
                    batch_size=batch_size
                ),
                feedforward=FeedForwardConfig(
                    proj_factor=xlstm_proj_factor,
                    act_fn="gelu",
                ),
            ),
            embedding_dim=hidden,
            num_blocks=xlstm_num_blocks,
            context_length=None,
            slstm_at=1,      
            mlstm_block=None 
        )

        self.backbone = xLSTMBlockStack(cfg)

        self.decoder = nn.Linear(hidden, num_stations * horizon)
        self.dropout = nn.Dropout(dropout)
    

    def forward(self, x):
        B, T, N, F = x.shape
        
        if self.add_storage:
            L_dir = build_advection_operator(A=self.A, add_storage=self.add_storage, learn_stor=self.learn_stor)

        x = x.reshape(B, T, N * F)
        x = self.embed(x)
        x = x.transpose(0, 1)  

        out = self.backbone(x)
        last = out[-1]

        last = self.dropout(last)

        pred = self.decoder(last)

        pred = pred.view(B, N, self.horizon).permute(0, 2, 1)
        
        if self.add_storage:
            return pred, L_dir
        else:
            return pred

