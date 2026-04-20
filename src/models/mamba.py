try:
    from mamba_ssm import Mamba as Mamba_SSM
except ModuleNotFoundError:
    Mamba_SSM = None
import torch
import torch.nn as nn
from .base_model import BaseModel


class Mamba(BaseModel):
    def __init__(self, num_stations: int, in_features: int, hidden_size: int, output_dim: int, horizon: int, cfg, mamba_ssm=None, dropout=0.0):
        super().__init__()
        self.num_stations = num_stations
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.horizon = horizon
        self.cfg = cfg
        self.mamba = Mamba_SSM(
                d_model=self.hidden_size,
                d_state=self.cfg["mamba_d_state"],
                d_conv=self.cfg["mamba_d_conv"],
                expand=self.cfg["mamba_expand"],
            ) 
        self.transition_layer = nn.Linear(num_stations * in_features, hidden_size)  # project input features to hidden
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, horizon * num_stations)  # directly output horizon steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:        
            """
            Expects shape (B, T, N, F)
            Returns:
                y_hat: (B, horizon, N, output_dim)
            """
            B, T, N, F = x.shape

            x_flat = x.reshape(B, T, N * F)          

            h = self.transition_layer(x_flat)       

            mamba_out = self.mamba(h)                

            last_hidden = mamba_out[:, -1, :]        

            out_flat = self.output_layer(self.dropout(last_hidden))  
            y_hat = out_flat.view(B, N, self.horizon).permute(0, 2, 1)

            return y_hat
