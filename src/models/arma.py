import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from joblib import Parallel, delayed


def ARMA(y_train: np.ndarray, 
        y_test: np.ndarray, 
        horizon: int, 
        order=(1, 0, 0)):
    """
    ARMA forecast per station with single fit per station.
    Rolling forecasts simulated by feeding predictions forward.
    Inputs
    ------
    y_train : (T_train, N)
    y_test  : (T_test, N)
    horizon : int
    order   : tuple (p,d,q)
    Outputs
    -------
    preds   : (B, H, N)
    """

    T_train, N = y_train.shape
    T_test = y_test.shape[0]
    H = horizon
    B_windows = T_test - H + 1  # rolling windows

    preds = np.zeros((B_windows, H, N))

    def forecast_station(station_idx):
        y_full = np.concatenate([y_train[:, station_idx], y_test[:, station_idx]])
        model = ARIMA(y_train[:, station_idx], order=order)
        fit = model.fit(method_kwargs={"warn_convergence": False, "disp": 0})
        forecasts = np.zeros((B_windows, H))

        # rolling forecasts
        for b in range(B_windows):
            start_idx = T_train + b - 1
            forecast = fit.forecast(steps=H)
            forecasts[b, :] = forecast
        return forecasts  # shape (B_windows, H)

    # Parallel across stations
    preds_list = Parallel(n_jobs=-1)(
        delayed(forecast_station)(i) for i in range(N)
    )

    preds = np.stack(preds_list, axis=-1)  # shape (B_windows, H, N)
    return preds, {}
