# Code for Physics-Informed RNNs via Directed Graph Laplacian Regularization for River Network Flow Forecasting


We propose a regularization technique for streamflow forecasting that adds learnable storage within a graph-directed Laplacian framework. This approach ensures mass conservation on the forecasted values.


### Installation

```bash
git clone https://github.com/pilotdt/StreamFlow-DirLap-MC.git

cd StreamFlow-DirLap-MC
```


###  Train the model



#### Train the model with regularization

Each model has its own .yaml configuration in ```/config```.  Any default parameters can be overriden and specified in the command line:


e.g.,

```bash

python3 src/main_<backbone>.py reg_4_loss=L_dir add_storage=True 

```

### Data 

Publicly available datasets are used:

- **ERA5-Land** (ECMWF): High-resolution land surface reanalysis data used as meteorological forcings [1].

- **CAMELS-GB v2**: A large-sample hydrology dataset for Great Britain providing catchment attributes and hydro-meteorological time series [2].

<br>
<br>

### Data References

[1] Muñoz Sabater, J. (2019). *ERA5-Land hourly data from 1981 to present*. Copernicus Climate Change Service (C3S).

[2] Coxon, G., Freer, J., Lane, R., et al. (2020). *CAMELS-GB: Hydrometeorological time series and landscape attributes for 671 catchments in Great Britain*. Earth System Science Data, 12, 2459–2483.
