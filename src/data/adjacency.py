import pandas as pd
import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg
import os
import pickle


def load_adjacency_matrix(csv_path, device=None, specific_order=None):
    ext = os.path.splitext(csv_path)[1]

    if ext == ".csv":
        df = pd.read_csv(csv_path)

        if specific_order is not None:
            station_order = specific_order
            # Reorder rows
            df[df.columns[0]] = df[df.columns[0]].astype(str).str.strip()
            df = df.set_index(df.columns[0]).loc[station_order]
            df = df[station_order]
        # Extract ids
        row_ids = df.index.values.astype(int)
        col_ids = df.columns.values.astype(int)
        # Adjacency matrix
        A = df.to_numpy(dtype=np.float32)

    elif ext == ".pkl":
        with open(csv_path, "rb") as f:
            df = pickle.load(f, encoding="latin1")   
        if isinstance(df, list):
            sensor_ids, _, adj_mx = df
            A = adj_mx.astype(np.float32)
            row_ids = np.array(sensor_ids).astype(int)
            col_ids = row_ids
    
    # Convert to torch tensor
    A_tensor = torch.from_numpy(A)
    if device is not None:
        A_tensor = A_tensor.to(device)
    return A_tensor, row_ids, col_ids


def adjacency_to_edge_index(A, row_ids, col_ids):
    src, dst = np.nonzero(A)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(A[src, dst], dtype=torch.float32)
    return edge_index, edge_weight


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.where(d > 0, np.power(d, -0.5), 0.0).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d[d == 0] = 1e-10
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=False):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)
