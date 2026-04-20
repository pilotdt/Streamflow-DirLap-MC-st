import numpy as np


def create_windows(X_full, flow_only, history, horizon):
    T, N, F = X_full.shape
    X_list, y_list = [], []

    for t in range(T - history - horizon + 1):
        X_list.append(X_full[t : t + history])                         # (history, N, F)
        y_list.append(flow_only[t + history : t + history + horizon])  # (horizon, N)

    X = np.stack(X_list)
    y = np.stack(y_list)
    return X, y


def create_windows_4_wbe(X_full, history, horizon):
    T, N, F = X_full.shape
    X_list = []

    for t in range(T - history - horizon + 1):
        X_list.append(X_full[t : t + history])                         # (history, N, F)

    X = np.stack(X_list)
    return X