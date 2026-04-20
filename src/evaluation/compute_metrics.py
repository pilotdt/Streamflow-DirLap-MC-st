import numpy as np
import pandas as pd
from .metrics import (
    mae, rmse, nse,
    nse_per_station,
    mae_high_low,
    mean_absolute_percentage_peak_error,
    mean_peak_timing,
    peak_time_bias,
    missed_peaks,
    false_alarm_rate,
    peak_metric_overall
)

def compute_avg_metrics(trues: np.ndarray, preds: np.ndarray) -> dict:
    """
    Compute all evaluation metrics.

    Parameters
    ----------
    trues : np.ndarray
        Ground truth values, shape (B, H, F)
    preds : np.ndarray
        Predicted values, shape (B, H, F)

    Returns
    -------
    dict
        Dictionary of metric_name -> value
    """
    preds_flat = preds.flatten()
    trues_flat = trues.flatten()
    return {
        'MAE':          mae(preds_flat, trues_flat),
        'RMSE':         rmse(preds_flat, trues_flat),
        'NSE':          nse_per_station(trues, preds),
        'MAE_HIGH':     mae_high_low(preds, trues)[0],
        'MAPE_PEAK':    peak_metric_overall(trues, preds, mean_absolute_percentage_peak_error),
        'MeanPeakTime': peak_metric_overall(trues, preds, mean_peak_timing),
        'PeakTimeBias': peak_metric_overall(trues, preds, peak_time_bias),
        'MissedPeaks':  peak_metric_overall(trues, preds, missed_peaks),
        'FAR':          peak_metric_overall(trues, preds, false_alarm_rate)
    }

def compute_avg_metrics_fast(trues: np.ndarray, preds: np.ndarray) -> dict:
    """
    Compute all evaluation metrics.

    Parameters
    ----------
    trues : np.ndarray
        Ground truth values, shape (B, H, F)
    preds : np.ndarray
        Predicted values, shape (B, H, F)

    Returns
    -------
    dict
        Dictionary of metric_name -> value
    """
    preds_flat = preds.flatten()
    trues_flat = trues.flatten()
    return {
        'MAE':          mae(preds_flat, trues_flat),
        'RMSE':         rmse(preds_flat, trues_flat)
    }

def compute_station_metrics(trues: np.ndarray, preds: np.ndarray):
    """
    Compute metrics per station.
    
    y_true, y_pred: torch.Tensor of shape [batch, seq_len, num_stations]
    """
    trues = trues.float()
    preds = preds.float()
    
    y_true_flat = trues.reshape(-1, trues.shape[2])  # [batch*horizon, num_stations]
    y_pred_flat = preds.reshape(-1, preds.shape[2])  # [batch*horizon, num_stations]

    metrics = []
    for i in range(y_true_flat.shape[1]):
        y_t = y_true_flat[:, i]
        y_p = y_pred_flat[:, i]
        
        metrics.append({
            "station": i,
            'MAE':          mae(y_p, y_t),
            'RMSE':         rmse(y_p, y_t),
            'NSE':          nse(y_p, y_t),
            'MAE_HIGH':     mae_high_low(y_p, y_t)[0],
            'MAPE_PEAK':    peak_metric_overall(trues, preds, mean_absolute_percentage_peak_error, mode='per_station'),
            'MeanPeakTime': peak_metric_overall(trues, preds, mean_peak_timing, mode='per_station'),
            'PeakTimeBias': peak_metric_overall(trues, preds, peak_time_bias, mode='per_station'),
            'MissedPeaks':  peak_metric_overall(trues, preds, missed_peaks, mode='per_station'),
            'FAR':          peak_metric_overall(trues, preds, false_alarm_rate, mode='per_station'),
        })
    
    df_metrics = pd.DataFrame(metrics)
    return df_metrics
