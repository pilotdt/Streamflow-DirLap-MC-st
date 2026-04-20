import torch


def loader_2_dcrnn_fmt(loader):
    if not isinstance(loader, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor")
    
    B, T, N, F = loader.shape
    loader_md = loader.permute(1, 0, 2, 3)
    loader_md = loader_md.reshape(T, B, N * F)
    
    return loader_md

def dcrnn_fmt_2_loader(outputs, num_stations, output_dim=1):
    horizon, batch_size, _ = outputs.shape
    outputs = outputs.reshape(horizon, batch_size, num_stations, output_dim)
    outputs = outputs.permute(1, 0, 2, 3)  
    return outputs