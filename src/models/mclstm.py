import torch
import torch.nn as nn
from typing import Tuple
from .base_model import BaseModel 


class MCLSTM(BaseModel):
    def __init__(self, num_stations: int, features_per_station: int, hidden_size: int, horizon: int, initial_forget_bias: float = None):
        super(MCLSTM, self).__init__()
        
        self.num_stations = num_stations
        self.features_per_station = features_per_station
        self.horizon = horizon
        
        n_mass_inputs = num_stations 
        n_aux_inputs = num_stations * (features_per_station - 1)
        
        self.mclstm = _MCLSTMCell(
            mass_input_size=n_mass_inputs,
            aux_input_size=n_aux_inputs,
            hidden_size=hidden_size,
            initial_forget_bias=initial_forget_bias
        )
        self.head = nn.Linear(hidden_size - 1, num_stations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, F = x.shape
        
        mass_indices = torch.arange(0, N * F, F, device=x.device)
        
        all_indices = torch.arange(0, N * F, device=x.device)
        mask = torch.ones(N * F, dtype=torch.bool, device=x.device)
        mask[mass_indices] = False
        aux_indices = all_indices[mask]

        x_flat = x.reshape(B, T, -1).transpose(0, 1) 
        x_m = x_flat[:, :, mass_indices]             
        x_a = x_flat[:, :, aux_indices]              

        m_out, c = self.mclstm(x_m, x_a)

        curr_c = c[-1] 
        preds = []
        
        for h in range(self.horizon):
            xt_m_future = torch.zeros_like(x_m[0])
            xt_a_future = x_a[-1] 
            
            mt_out, curr_c = self.mclstm._step(xt_m_future, xt_a_future, curr_c)
            
            y_hat = mt_out[:, 1:]
            y_hat = self.head(y_hat)
            preds.append(y_hat)

        return torch.stack(preds, dim=1)

class _MCLSTMCell(nn.Module):
    def __init__(self, mass_input_size, aux_input_size, hidden_size, initial_forget_bias=None):
        super(_MCLSTMCell, self).__init__()
        self._hidden_size = hidden_size
        gate_inputs = aux_input_size + hidden_size + mass_input_size

        self.output_gate = _Gate(in_features=gate_inputs, out_features=hidden_size)
        self.input_gate = _NormalizedGate(in_features=gate_inputs, out_shape=(mass_input_size, hidden_size), normalizer="normalized_sigmoid")
        self.redistribution = _NormalizedGate(in_features=gate_inputs, out_shape=(hidden_size, hidden_size), normalizer="normalized_relu")

        if initial_forget_bias is not None:
            nn.init.constant_(self.output_gate.fc.bias, val=initial_forget_bias)

    def forward(self, x_m, x_a):
        _, batch_size, _ = x_m.size()
        ct = x_m.new_zeros((batch_size, self._hidden_size))
        m_out, c = [], []
        for xt_m, xt_a in zip(x_m, x_a):
            mt_out, ct = self._step(xt_m, xt_a, ct)
            m_out.append(mt_out)
            c.append(ct)
        return torch.stack(m_out), torch.stack(c)

    def _step(self, xt_m, xt_a, c):
        features = torch.cat([xt_m, xt_a, c / (c.norm(1, dim=-1, keepdim=True) + 1e-5)], dim=-1)
        i, r, o = self.input_gate(features), self.redistribution(features), self.output_gate(features)
        m_in = torch.matmul(xt_m.unsqueeze(-2), i).squeeze(-2)
        m_sys = torch.matmul(c.unsqueeze(-2), r).squeeze(-2)
        m_new = m_in + m_sys
        return o * m_new, (1 - o) * m_new

class _Gate(nn.Module):
    """Utility class to implement a standard sigmoid gate"""

    def __init__(self, in_features: int, out_features: int):
        super(_Gate, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through the normalised gate"""
        return torch.sigmoid(self.fc(x))

class _NormalizedGate(nn.Module):
    def __init__(self, in_features: int, out_shape: Tuple[int, int], normalizer: str):
        super(_NormalizedGate, self).__init__()
        self.out_shape = out_shape
        
        bottleneck_dim = 256 
        
        self.fc = nn.Sequential(
            nn.Linear(in_features, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, out_shape[0] * out_shape[1])
        )

        if normalizer == "normalized_sigmoid":
            self.activation = nn.Sigmoid()
        elif normalizer == "normalized_relu":
            self.activation = nn.ReLU()
        
        self._reset_parameters()

    def _reset_parameters(self):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x).view(-1, *self.out_shape)
        return torch.nn.functional.normalize(self.activation(h), p=1, dim=-1)