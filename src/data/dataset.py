import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import List, Tuple
from statsmodels.tsa.stattools import pacf


class RiverFlowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_flow_csv(flow_csv: str) -> Tuple[pd.DatetimeIndex, pd.DataFrame]:
    df = pd.read_csv(flow_csv, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    dates = pd.DatetimeIndex(df['date'])
    flow_df = df.drop(columns=['date'])
    return dates, flow_df


def load_other_csv(file_path: str) -> Tuple[pd.DatetimeIndex, pd.DataFrame]:
    """
    Loads PEMS-BAY or METR-LA traffic datasets from a CSV file.
    Expects a 'date' column or the first column to be the timestamp.
    """
    df = pd.read_csv(file_path)

    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'date'})
    
    df['date'] = pd.to_datetime(df['date'])
    
    df = df.sort_values('date').reset_index(drop=True)
    
    dates = pd.DatetimeIndex(df['date'])
    
    data_df = df.drop(columns=['date'])
    data_df.ffill(inplace=True)
    data_df.bfill(inplace=True)
    return dates, data_df



def load_clim_csv(clim_csv: str, station_names: List[str], dates: pd.DatetimeIndex):
    """
    Load and align climatic data with river flow dates and station IDs.
    Ensures:
        - Only stations present in flow_df are used
        - Climate dates are truncated to flow dates
        - Output shape: (B, N, F)
    """
    clim = pd.read_csv(clim_csv, parse_dates=['date'])
    clim['date'] = pd.to_datetime(clim['date'])
    clim = clim.sort_values('date')

    clim = clim[clim['date'].isin(dates)]
    clim['id'] = clim['id'].astype(str)
    clim = clim.drop(columns=[c for c in clim.columns if 'Unnamed' in c])
    var_cols = [c for c in clim.columns if c not in ('date', 'id')]

    unique_clim_ids = set(clim['id'].unique())
    flow_ids = set(station_names)

    missing_ids = flow_ids - unique_clim_ids
    if missing_ids:
        print(f"WARNING: {len(missing_ids)} flow stations have NO climate data:")
        print(sorted(missing_ids)[:20], "...")

    unique_clim_ids2 = clim['id'].unique()
    flow_ids2 = station_names
    clim = clim[clim['id'].isin(flow_ids2)]
    order = clim['id'].astype(str).unique().tolist()
    
    clim = clim.set_index('id').reset_index()

    T = len(dates)
    N = len(order)
    F = len(var_cols)

    out = np.full((T, N, F), np.nan, dtype=float)
    name2idx = {name: i for i, name in enumerate(order)}
    date_index = {d: i for i, d in enumerate(dates)}

    for row in clim.itertuples(index=False):
        d = row.date
        sid = row.id

        if d not in date_index:
            continue  

        t_idx = date_index[d]
        s_idx = name2idx[sid]

        values = np.array([getattr(row, v) for v in var_cols], dtype=float)
        out[t_idx, s_idx, :] = values

    for i in range(N):
        for f in range(F):
            ts = pd.Series(out[:, i, f], index=dates)
            # ts = ts.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
            ts = ts.ffill().bfill().fillna(0.0)
            out[:, i, f] = ts.values

    return out, var_cols, order


def load_station_attributes(
    attrs_csv: str,
    station_names: list,
    attr_cols: list
):
    """
    Load static station attributes and align them to the station order in flow_df.

    Output:
        static_arr: (N, S)
        attr_cols_filtered: list of columns actually used
    """
    attrs = pd.read_csv(attrs_csv)
    attrs['id'] = attrs['id'].astype(str)
    attrs = attrs.drop(columns=[c for c in attrs.columns if 'Unnamed' in c])
    # Ensure 'id' exists
    if 'id' not in attrs.columns:
        raise ValueError("Station attributes CSV must contain an 'id' column.")

    # Filter only required columns
    missing = [c for c in attr_cols if c not in attrs.columns]
    if missing:
        print("WARNING: The following attribute columns are missing:", missing)

    attr_cols = [c for c in attr_cols if c in attrs.columns]
    # Keep only stations that appear in flow_df
    attrs = attrs[attrs['id'].isin(station_names)]
    attrs = attrs.set_index('id').loc[station_names].reset_index()

    # Warn about missing stations
    missing_stations = set(station_names) - set(attrs['id'])
    if missing_stations:
        print("WARNING: These flow stations have NO static attributes:")
        print(sorted(list(missing_stations))[:20], "...")

    attrs = attrs.set_index('id')

    N = len(station_names)
    S = len(attr_cols)

    out = np.zeros((N, S), dtype=float)

    for i, name in enumerate(station_names):
        if name in attrs.index:
            out[i] = attrs.loc[name, attr_cols].values
        else:
            out[i] = np.nan  
    
    df = pd.DataFrame(out, columns=attr_cols)
    df = df.fillna(df.mean()).fillna(0.0)

    return df.values.astype(float), attr_cols


def estimate_lookback_pacf(X, max_lag=90, min_lag=1, threshold=0.15):
    """
    Estimate optimal lookback length per station using partial autocorrelation (PACF).

    Args:
        X: np.ndarray of shape (T, N, 1) - past river flow data
        max_lag: maximum lag to consider
        min_lag: minimum lag to return
        threshold: PACF value to consider significant

    Returns:
        max_lookback: int, maximum lookback across basins
        lookbacks_per_basin: np.ndarray of shape (N,) optimal lookback per basin
    """
    T, N = X.shape

    lookbacks_per_basin = np.zeros(N, dtype=int)

    for basin in range(N):
        series = X[:, basin]
        pacf_vals = pacf(series, nlags=max_lag, method='ywm')
        significant_lags = np.where(np.abs(pacf_vals[1:]) > threshold)[0] + 1 
        if len(significant_lags) == 0:
            best_lag = min_lag
        else:
            best_lag = max(significant_lags)
        lookbacks_per_basin[basin] = max(best_lag, min_lag)

    max_lookback = int(lookbacks_per_basin.max())
    return max_lookback, lookbacks_per_basin

