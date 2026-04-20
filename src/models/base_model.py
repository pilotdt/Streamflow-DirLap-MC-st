import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Abstract base class for forecasting models.
    DL models should inherit from this class.
    """
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, x):
        """
        Function for inference.
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)
