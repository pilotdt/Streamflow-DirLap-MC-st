
import os
import yaml
import torch
import numpy as np
from .dataset import load_flow_csv, load_clim_csv, load_station_attributes, load_other_csv
from .dataset import estimate_lookback_pacf
from .windowing import create_windows
# , create_windows_4_wbe
from training.utils import StandardScaler, MinMaxScaler
import pickle


DATA_KEYS = [
    "X_train", "y_train",
    "X_val", "y_val",
    "X_test", "y_test", "clim_station_order",
    "num_stations", "in_features",
    "dates", "y_scaler",
    "std_per_station",
    "seq_lengths"
]

class DataPreparer:
    def __init__(self, cfg, args=None):
        self.cfg = cfg
        self.args = args

    def prepare_data(self):
        """Load and prepare data for training"""
        if self.cfg.get('other_csv')=='':
            dates, flow_df = load_flow_csv(self.cfg['flow_csv'])
            flow_df = flow_df.drop(columns='Unnamed: 0')
            flow_df.columns = flow_df.columns.astype(str)   
            if self.cfg.get('clim_csv'):
                clim_data, clim_vars, clim_station_order = load_clim_csv(
                    self.cfg['clim_csv'],
                    station_names=list(flow_df.columns),
                    dates=dates
                )
            else:
                clim_data = None
                clim_station_order = list(flow_df.columns)    
            T = len(dates)
            N = len(clim_station_order)
            flow_df = flow_df[clim_station_order]
            flow_arr = flow_df.values[:, :, None]
            if self.cfg.get("station_attrs"):
                static_attrs, static_attr_cols = load_station_attributes(
                    self.cfg["station_attrs"],
                    station_names=clim_station_order, 
                    attr_cols=self.cfg["station_attrs_cols"]
                )
                static_broadcast = np.broadcast_to(static_attrs, (T, N, static_attrs.shape[1]))
            else:
                static_broadcast = None

            pieces = [flow_arr]                  
            if clim_data is not None:
                pieces.append(clim_data)         
            if static_broadcast is not None:
                pieces.append(static_broadcast)  
            
            X_full = np.concatenate(pieces, axis=2)
            y_full = flow_df.values  

            n = len(X_full)

        else:
            dates, other_df = load_other_csv(self.cfg['other_csv'])    
            X_full = other_df.values
            y_full = X_full
            n = len(X_full)
            clim_station_order = None

        train_end = int(self.cfg['train_ratio'] * n)
        val_end = train_end + int(self.cfg['val_ratio'] * n)

        X_train_raw = X_full[:train_end]
        X_val_raw   = X_full[train_end:val_end]
        X_test_raw  = X_full[val_end:]

        y_train_raw = y_full[:train_end]
        y_val_raw   = y_full[train_end:val_end]
        y_test_raw  = y_full[val_end:]


        ## X_test_4_wbe = create_windows_4_wbe(
        ##     X_full=X_test_raw,
        ##     history=max_lookb if self.cfg['use_packing'] else self.cfg['history'],
        ##     horizon=self.cfg['horizon']
        ## )
        
        # X_train_4_wbe = create_windows_4_wbe(
        #     X_full=X_train_raw,
        #     history=max_lookb if self.cfg['use_packing'] else self.cfg['history'],
        #     horizon=self.cfg['horizon']
        # )

        # X_val_4_wbe = create_windows_4_wbe(
        #     X_full=X_val_raw,
        #     history=max_lookb if self.cfg['use_packing'] else self.cfg['history'],
        #     horizon=self.cfg['horizon']
        # )
        
        # if self.cfg['scaler'] is not None: 
        #     if self.cfg['scaler']=="StandardScaler":
        #         x_scaler = StandardScaler()
        #     elif self.cfg['scaler']=="MinMaxScaler":
        #         x_scaler = MinMaxScaler()
            
        #     x_scaler.fit(X_train_raw)
            
        #     X_train_4_wbe_scaled = x_scaler.transform(X_train_4_wbe)
        #     X_val_4_wbe_scaled   = x_scaler.transform(X_val_4_wbe)
        
        # print(X_train_4_wbe_scaled.shape)
        # np.save(f"data/X_train_4_wbe_scaled_10.npy", X_train_4_wbe_scaled)

        # print(X_val_4_wbe_scaled.shape)
        # np.save(f"data/X_val_4_wbe_scaled_10.npy", X_val_4_wbe_scaled)
        # print(a)

        ## print(X_test_4_wbe.shape)
        ## np.save(f"data/X_test_4_wbe_10.npy", X_test_4_wbe)
        ## print(a)
        
        if self.cfg['scaler'] is not None: 
            if self.cfg['scaler']=="StandardScaler":
                x_scaler = StandardScaler()
            elif self.cfg['scaler']=="MinMaxScaler":
                x_scaler = MinMaxScaler()
            
            x_scaler.fit(X_train_raw)
            
            X_train_scaled = x_scaler.transform(X_train_raw)
            X_val_scaled   = x_scaler.transform(X_val_raw)
            X_test_scaled  = x_scaler.transform(X_test_raw)

            if self.cfg['scaler']=="StandardScaler":
                y_scaler = StandardScaler()
                y_scaler.fit(y_train_raw) 
            elif self.cfg['scaler']=="MinMaxScaler":
                y_scaler = MinMaxScaler()
                y_scaler.fit(y_train_raw)  

            y_train_scaled = y_scaler.transform(y_train_raw)
            y_val_scaled   = y_scaler.transform(y_val_raw)
            y_test_scaled  = y_scaler.transform(y_test_raw)
        else:
            y_scaler = None
            x_scaler = None

        per_station_std = torch.tensor(
            np.nanstd(y_train_raw, axis=0),
            dtype=torch.float32
        )                                   
        per_station_std = per_station_std[None, None, :]
        if self.cfg['use_packing']==True:    
            max_lookb, lookbacks_per_basin = estimate_lookback_pacf(y_train_raw)
            seq_lengths = torch.tensor(lookbacks_per_basin, dtype=torch.int)
        else:
            seq_lengths = None

        X_train, y_train = create_windows(
            X_full= X_train_scaled if self.cfg['scaler'] is not None else X_train_raw,
            flow_only= y_train_scaled if self.cfg['scaler'] is not None else y_train_raw,
            history=max_lookb if self.cfg['use_packing'] else self.cfg['history'],
            horizon=self.cfg['horizon']
        )

        X_val, y_val = create_windows(
            X_full= X_val_scaled if self.cfg['scaler'] is not None else X_val_raw,
            flow_only= y_val_scaled if self.cfg['scaler'] is not None else y_val_raw,
            history=max_lookb if self.cfg['use_packing'] else self.cfg['history'],
            horizon=self.cfg['horizon']
        )

        X_test, y_test = create_windows(
            X_full= X_test_scaled if self.cfg['scaler'] is not None else X_test_raw,
            flow_only= y_test_scaled if self.cfg['scaler'] is not None  else y_test_raw,
            history=max_lookb if self.cfg['use_packing'] else self.cfg['history'],
            horizon=self.cfg['horizon']
        )


        if X_full.ndim == 2:
            X_full = np.expand_dims(X_full, axis=-1)
        
        num_stations = X_full.shape[1]
        in_features = X_full.shape[2]
        
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "clim_station_order": clim_station_order,
            "num_stations": num_stations,
            "in_features": in_features,
            "dates": dates,
            "y_scaler": y_scaler,
            "std_per_station": per_station_std,
            "seq_lengths": seq_lengths
        }

    def load_prepared_data(self, save_dir):
        if self.cfg.get('other_csv')=='':
            tensors = torch.load(os.path.join(save_dir, f"tensors-h={self.cfg['horizon']}-l={self.cfg['history']}-scaler-{self.cfg['scaler']}.pt"))
            with open(os.path.join(save_dir, f"objects-h={self.cfg['horizon']}-l={self.cfg['history']}-scaler-{self.cfg['scaler']}.pkl"), "rb") as f:
                objects = pickle.load(f)
        else:
            tensors = torch.load(os.path.join(save_dir, f"tensors-{(self.cfg['other_csv']).split('/')[1].split('.')[0]}-h={self.cfg['horizon']}-l={self.cfg['history']}-scaler-{self.cfg['scaler']}.pt"))
            with open(os.path.join(save_dir, f"objects-{(self.cfg['other_csv']).split('/')[1].split('.')[0]}-h={self.cfg['horizon']}-l={self.cfg['history']}-scaler-{self.cfg['scaler']}.pkl"), "rb") as f:
                objects = pickle.load(f)        
        data = {}
        data.update(tensors)
        data.update(objects)
        return data
    
    def save_prepared_data(self, save_dir, **kwargs):
        os.makedirs(save_dir, exist_ok=True)

        for k in DATA_KEYS:
            if k not in kwargs:
                raise KeyError(f"Missing key '{k}' in save_prepared_data")

        if self.cfg.get('other_csv')=='':
            torch.save(
                {k: kwargs[k] for k in DATA_KEYS if torch.is_tensor(kwargs[k])},
                os.path.join(save_dir, f"tensors-h={self.cfg['horizon']}-l={self.cfg['history']}-scaler-{self.cfg['scaler']}.pt")
            )
            with open(os.path.join(save_dir, f"objects-h={self.cfg['horizon']}-l={self.cfg['history']}-scaler-{self.cfg['scaler']}.pkl"), "wb") as f:
                pickle.dump(
                    {k: kwargs[k] for k in DATA_KEYS if not torch.is_tensor(kwargs[k])},
                    f
                )
        else:            
            torch.save(
                {k: kwargs[k] for k in DATA_KEYS if torch.is_tensor(kwargs[k])},
                os.path.join(save_dir, f"tensors-{(self.cfg['other_csv']).split('/')[1].split('.')[0]}-h={self.cfg['horizon']}-l={self.cfg['history']}-scaler-{self.cfg['scaler']}.pt")
            )
            with open(os.path.join(save_dir, f"objects-{(self.cfg['other_csv']).split('/')[1].split('.')[0]}-h={self.cfg['horizon']}-l={self.cfg['history']}-scaler-{self.cfg['scaler']}.pkl"), "wb") as f:
                pickle.dump(
                    {k: kwargs[k] for k in DATA_KEYS if not torch.is_tensor(kwargs[k])},
                    f
                )

def load_config(config_path):
    """Load configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


