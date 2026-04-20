import numpy as np
from scipy import signal
from scipy.signal import find_peaks
from xarray import DataArray


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))


def nse(y_true, y_pred):
    denom = np.sum((y_true - np.mean(y_true))**2)
    if denom == 0:
        return np.nan
    return 1 - np.sum((y_true - y_pred)**2) / denom


def nse_per_station(y_true, y_pred):
    if y_true.ndim == 2:  
        B, N = y_true.shape
        scores = []
        for n in range(N):
            obs = y_true[:, n]  
            sim = y_pred[:,n]
            if len(obs) == 0:
                continue
            denom = np.sum((obs - obs.mean())**2)
            if denom < 1e-8:
                continue
            nse = 1 - np.sum((obs - sim)**2) / denom
            scores.append(nse)
        return np.mean(scores)
    elif y_true.ndim == 3:

        B, H, N = y_true.shape
        scores = []
        for n in range(N):
            obs = y_true[:, :, n].reshape(-1)
            sim = y_pred[:, :, n].reshape(-1)
            if len(obs) == 0:
                continue
            denom = np.sum((obs - obs.mean())**2)
            if denom < 1e-8:
                continue
            nse = 1 - np.sum((obs - sim)**2) / denom
            scores.append(nse)
        return np.mean(scores)


def mae_high_low(y_true, y_pred, quantile=0.95):
    flat_true = y_true.flatten()
    mask_high = flat_true >= np.quantile(flat_true, quantile)
    mask_low = flat_true <= np.quantile(flat_true, 1-quantile)
    mae_h = mae(flat_true[mask_high], y_pred.flatten()[mask_high]) if mask_high.sum() > 0 else np.nan
    mae_l = mae(flat_true[mask_low], y_pred.flatten()[mask_low]) if mask_low.sum() > 0 else np.nan
    return mae_h, mae_l

def mean_absolute_percentage_peak_error(obs: DataArray, sim: DataArray) -> float:
    r"""Calculate the mean absolute percentage error (MAPE) for peaks

    .. math:: \text{MAPE}_\text{peak} = \frac{1}{P}\sum_{p=1}^{P} \left |\frac{Q_{s,p} - Q_{o,p}}{Q_{o,p}} \right | \times 100,

    where :math:`Q_{s,p}` are the simulated peaks (here, `sim`), :math:`Q_{o,p}` the observed peaks (here, `obs`) and
    `P` is the number of peaks.

    Uses scipy.find_peaks to find peaks in the observed time series. The observed peaks indices are used to subset
    observed and simulated flows. Finally, the MAPE metric is calculated as the mean absolute percentage error
    of observed peak flows and corresponding simulated flows.

    Parameters
    ----------
    obs : DataArray
        Observed time series.
    sim : DataArray
        Simulated time series.

    Returns
    -------
    float
        Mean absolute percentage error (MAPE) for peaks.
    """
    # return np.nan if there are no valid observed or simulated values
    if obs.size == 0 or sim.size == 0:
        return np.nan

    # heuristic to get indices of peaks and their corresponding height.
    peaks, _ = signal.find_peaks(obs, distance=100, prominence=np.std(obs))

    # check if any peaks exist, otherwise return np.nan
    if peaks.size == 0:
        return np.nan

    # subset data to only peak values
    obs = obs[peaks]
    sim = sim[peaks]

    # calculate the mean absolute percentage peak error
    peak_mape = np.sum(np.abs((sim - obs) / obs)) / peaks.size * 100

    return peak_mape


def peak_time_bias(y_true, y_pred, prominence=1.0):
    """
    Computes the Peak Time Bias.

    Measures whether a model consistently predicts peak timing too early or too late.

    Args:
        y_true (np.ndarray): Array of true flow values.
        y_pred (np.ndarray): Array of predicted flow values.
        prominence (float): Prominence threshold for peak detection to ignore noise.
                            A higher value means only more significant peaks are detected.

    Returns:
        float: The average time difference between predicted and true peaks.
               Returns np.nan if no peaks are found.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Find peak indices in both series. We use scipy's find_peaks for robustness.
    true_peak_indices, _ = find_peaks(y_true, prominence=prominence)
    pred_peak_indices, _ = find_peaks(y_pred, prominence=prominence)
    
    if len(true_peak_indices) == 0 or len(pred_peak_indices) == 0:
        return np.nan
    
    # We need to find the closest predicted peak for each true peak
    time_differences = []
    for true_peak_idx in true_peak_indices:
        # Find the index of the closest predicted peak in time
        closest_pred_idx = np.argmin(np.abs(pred_peak_indices - true_peak_idx))
        time_difference = pred_peak_indices[closest_pred_idx] - true_peak_idx
        time_differences.append(time_difference)
        
    if not time_differences:
      return np.nan

    return np.mean(time_differences)
    
def mean_peak_timing(obs: DataArray,
                     sim: DataArray,
                     window: int = 3,
                     resolution: str = '1D',
                     datetime_coord: str = 'date') -> float:
    """Mean difference in peak flow timing.
    
    Uses scipy.find_peaks to find peaks in the observed time series. Starting with all observed peaks, those with a
    prominence of less than the standard deviation of the observed time series are discarded. Next, the lowest peaks
    are subsequently discarded until all remaining peaks have a distance of at least 100 steps. Finally, the
    corresponding peaks in the simulated time series are searched in a window of size `window` on either side of the
    observed peaks and the absolute time differences between observed and simulated peaks is calculated.
    The final metric is the mean absolute time difference across all peaks. For more details, see Appendix of [#]_
    
    Parameters
    ----------
    obs : DataArray
        Observed time series.
    sim : DataArray
        Simulated time series.
    window : int, optional
        Size of window to consider on each side of the observed peak for finding the simulated peak. That is, the total
        window length to find the peak in the simulations is :math:`2 * \\text{window} + 1` centered at the observed
        peak. The default depends on the temporal resolution, e.g. for a resolution of '1D', a window of 3 is used and 
        for a resolution of '1h' the the window size is 12.
    resolution : str, optional
        Temporal resolution of the time series in pandas format, e.g. '1D' for daily and '1h' for hourly.
    datetime_coord : str, optional
        Name of datetime coordinate. Tried to infer automatically if not specified.
        

    Returns
    -------
    float
        Mean peak time difference.

    References
    ----------
    .. [#] Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple 
        meteorological datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss., 
        https://doi.org/10.5194/hess-2020-221, in review, 2020. 
    """
    obs = np.asarray(obs)
    sim = np.asarray(sim)

    peaks, _ = signal.find_peaks(obs, distance=100, prominence=np.std(obs))

    timing_errors = []

    for idx in peaks:
        if idx - window < 0 or idx + window >= len(obs):
            continue

        window_vals = sim[idx - window: idx + window + 1]
        sim_peak_offset = np.argmax(window_vals)
        sim_peak_idx = idx - window + sim_peak_offset

        timing_errors.append(abs(sim_peak_idx - idx))

    return np.mean(timing_errors) if timing_errors else np.nan

    
def missed_peaks(obs: DataArray,
                 sim: DataArray,
                 window: int = 1,
                 resolution: str = '1D',
                 percentile: float = 80,
                 datetime_coord: str = 'date') -> float:
    """Fraction of missed peaks.
    
    Uses scipy.find_peaks to find peaks in the observed and simulated time series above a certain percentile. Counts
    the number of peaks in obs that do not exist in sim within the specified window.

    Parameters
    ----------
    obs : DataArray
        Observed time series.
    sim : DataArray
        Simulated time series.
    window : int, optional
        Size of window to consider on each side of the observed peak for finding the simulated peak. That is, the total
        window length to find the peak in the simulations is :math:`2 * \\text{window} + 1` centered at the observed
        peak. The default depends on the temporal resolution, e.g. for a resolution of '1D', a window of 1 is used and 
        for a resolution of '1h' the the window size is 12. Note that this is a different default window size than is
        used in the peak-timing metric for '1D'.
    resolution : str, optional
        Temporal resolution of the time series in pandas format, e.g. '1D' for daily and '1h' for hourly.
    percentile: float, optional
        Only consider peaks above this flow percentile (0, 100).
    datetime_coord : str, optional
        Name of datetime coordinate. Tried to infer automatically if not specified.

    Returns
    -------
    float
        Fraction of missed peaks.   
    """
    obs = np.asarray(obs)
    sim = np.asarray(sim)

    if obs.size == 0 or sim.size == 0:
        return np.nan

    # Peak height thresholds
    min_obs_height = np.percentile(obs, percentile)
    min_sim_height = np.percentile(sim, percentile)

    # Find peaks
    peaks_obs, _ = signal.find_peaks(obs, height=min_obs_height, distance=30)
    peaks_sim, _ = signal.find_peaks(sim, height=min_sim_height, distance=30)

    if len(peaks_obs) == 0:
        return 0.0

    missed_events = 0

    for idx in peaks_obs:
        # Skip peaks too close to boundaries
        if idx - window < 0 or idx + window >= len(obs):
            continue

        # Check for any simulated peak within ±window
        if not np.any(np.abs(peaks_sim - idx) <= window):
            missed_events += 1

    return missed_events / len(peaks_obs)


def false_alarm_rate(y_true, y_pred, threshold=0.95):
    """
    Computes the False Alarm Rate (FAR).

    FAR is the proportion of times a model predicts a peak that doesn't occur.

    Args:
        y_true (np.ndarray): Array of true flow values.
        y_pred (np.ndarray): Array of predicted flow values.
        threshold (float): A value above which a prediction is considered a "peak."

    Returns:
        float: The FAR score. Returns np.nan if there are no predicted peaks.
    """
    y_true = np.asarray(y_true)
    threshold = np.quantile(y_true, threshold)
    y_pred = np.asarray(y_pred)
    
    # A peak event is a value above the threshold
    true_peaks = y_true > threshold
    pred_peaks = y_pred > threshold
    
    # True positives (correct predictions)
    correct_predictions = np.sum(np.logical_and(true_peaks, pred_peaks))
    
    # False alarms (predicted peak, no true peak)
    false_alarms = np.sum(np.logical_and(~true_peaks, pred_peaks))
    
    denominator = false_alarms + correct_predictions
    
    if denominator == 0:
        return np.nan
    
    far = false_alarms / denominator
    return far


def peak_metric_overall(y_true, y_pred, metric_fn, mode='avg'):
    """
    Compute metric over peaks for either:
        - 'avg': overall average across all stations and batches
        - 'per_station': per-station average across batches
    
    Args:
        y_true, y_pred: np.array or torch.Tensor of shape (B, H, F)
        metric_fn: function expecting 1D arrays (obs, sim)
        mode: 'avg' or 'per_station'
    
    Returns:
        float (if mode='avg') or array of shape (F,) (if mode='per_station')
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # --- Ensure shape is (B, H, F) ---
    if y_true.ndim == 2:
        T, F = y_true.shape

        if mode == 'avg':
            scores = []
            for f in range(F):
                obs = y_true[:, f]  
                sim = y_pred[:, f]

                if np.all(np.isnan(obs)):
                    continue

                val = metric_fn(obs, sim)
                if not np.isnan(val):
                    scores.append(val)

            return np.mean(scores) if scores else np.nan

        elif mode == 'per_station':
            per_station_scores = []
            for f in range(F):
                obs = y_true[:, f]
                sim = y_pred[:, f]

                if np.all(np.isnan(obs)):
                    per_station_scores.append(np.nan)
                    continue

                val = metric_fn(obs, sim)
                per_station_scores.append(val if not np.isnan(val) else np.nan)

            return np.array(per_station_scores)

        else:
            raise ValueError(f"Invalid mode: {mode}")
    elif y_true.ndim == 3:
        B, H, F = y_true.shape

        if mode == 'avg':
            scores = []
            for b in range(B):
                for f in range(F):
                    obs = y_true[b, :, f]
                    sim = y_pred[b, :, f]

                    if np.all(np.isnan(obs)):
                        continue

                    val = metric_fn(obs, sim)
                    if not np.isnan(val):
                        scores.append(val)
            return np.mean(scores) if scores else np.nan

        elif mode == 'per_station':
            per_station_scores = []
            for f in range(F):
                station_scores = []
                for b in range(B):
                    obs = y_true[b, :, f]
                    sim = y_pred[b, :, f]
                    if np.all(np.isnan(obs)):
                        continue
                    val = metric_fn(obs, sim)
                    if not np.isnan(val):
                        station_scores.append(val)
                per_station_scores.append(np.mean(station_scores) if station_scores else np.nan)
            return np.array(per_station_scores)

        else:
            raise ValueError(f"Invalid mode: {mode}. Choose 'avg' or 'per_station'.")


