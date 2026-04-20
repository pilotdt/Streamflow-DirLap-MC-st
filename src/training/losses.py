import torch
import torch.nn as nn


class MSELoss(nn.Module):

    def __init__(self, eps=1e-6, use_rmse=False):
        super().__init__()
        self.eps = eps
        self.use_rmse = use_rmse

    def forward(self, y_hat, y):
            mask = ~torch.isnan(y)

            scaled_se = ((y_hat - y)**2)

            valid_se = scaled_se[mask]

            if self.use_rmse:
                return torch.sqrt(valid_se.mean())
            else:
                return valid_se.mean()


class StdScaledMSELoss(nn.Module):
    """
    RMSE or MSE scaled by per-station standard deviation.
    """

    def __init__(self, eps=1e-6, use_rmse=False):
        super().__init__()
        self.eps = eps
        self.use_rmse = use_rmse

    def forward(self, y_hat, y, per_station_std):
            mask = ~torch.isnan(y)
            device = y_hat.device
            if isinstance( per_station_std, torch.Tensor):
                per_station_std =  per_station_std.to(device)
            else:
                per_station_std = torch.tensor(per_station_std, device=device)

            std = per_station_std.expand_as(y_hat)

            scaled_se = ((y_hat - y)**2) / ((std + self.eps)**2)

            valid_se = scaled_se[mask]

            if self.use_rmse:
                return torch.sqrt(valid_se.mean())
            else:
                return valid_se.mean()
