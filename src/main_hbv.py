"""
HBV Hydrologic model
Example usage: python3 src/main_hbv.py
"""

import hydra
from omegaconf import DictConfig
import os
import gc
import yaml
import json
import numpy as np
import pandas as pd
import time

from data.dataloader import DataPreparer
from rrmpg.models import HBVEdu  
from rrmpg.utils.metrics import calc_nse
from tqdm.notebook import tqdm
from utils.seed import set_seed
from utils.logger import get_logger
from evaluation.compute_metrics import compute_avg_metrics
from joblib import Parallel, delayed

def compute_monthly_means(series, months):
    return np.stack([
        series[months == m].mean(axis=0) for m in range(1, 13)
    ], axis=1)  

def process_single_station(n,qobs, prec, temp, month_train, pet_m, temp_m, 
                        prec_test, temp_test, month_test, pet_m_test, temp_m_test, qobs_test, horizon):
    
    model = HBVEdu()

    # print(n)
    # print(qobs[0])
    # print(prec[0])
    # print(temp[0])
    # print(np.unique(month_train))
    # print(pet_m[0])
    # print(temp_m[0])
    train_start = time.perf_counter()
    result = model.fit(
        qobs, temp, prec, 
        month_train, pet_m, temp_m
    )
    # print(result)
    # print(result.x)
    train_time = time.perf_counter() - train_start
    param_per_station = dict(zip(model.get_parameter_names(), result.x))
    # print(param_per_station)
    model.set_params(param_per_station)
    infer_start = time.perf_counter()
    qsim_continuous = model.simulate(
        temp_test, prec_test,
        month_test, pet_m_test, temp_m_test
    )
    infer_time = time.perf_counter() - infer_start

    nse_fit = calc_nse(qobs_test, qsim_continuous)
    
    return qsim_continuous, qobs_test, train_time, infer_time, param_per_station, nse_fit

def run_single_experiment(cfg, 
                        run_id, 
                        num_stations,
                        logger, 
                        prec_test,
                        temp_test,
                        month_test,
                        pet_m_test,
                        temp_m_test,
                        prec,
                        temp,
                        month_train,
                        pet_m,
                        temp_m,
                        qobs,
                        qobs_test
                        ):
    
    run_dir = os.path.join(cfg["results_dir"], f"run_{run_id + 1}")
    os.makedirs(run_dir, exist_ok=True)
    seed = cfg["start_seed"] + run_id
    set_seed(seed)
    set_seed(cfg['start_seed'] + run_id)
    logger.info(f"\n{'='*60}")
    logger.info(f"Run {run_id+1}/{cfg['num_runs']} - Seed: {seed}")
    logger.info(f"{'='*60}")

    train_times = []
    infer_times = []
    preds_all = []
    trues_all = []
    station_params = {}

    results = Parallel(n_jobs=-1)(
    delayed(process_single_station)(n,
        qobs[:,n], prec[:,n], temp[:,n], month_train, pet_m[n], temp_m[n],
        prec_test[:,n], temp_test[:,n], month_test, pet_m_test[n], temp_m_test[n], qobs_test[:,n], cfg["horizon"]
    ) for n in tqdm(range(num_stations), desc="Processing HBV per site"))
    
    preds_all = [r[0] for r in results]
    trues_all = [r[1] for r in results]
    train_times = [r[2] for r in results]
    infer_times = [r[3] for r in results]
    station_params = {station_id: r[4] for station_id, r in zip(range(num_stations), results)}
    nses = [r[5] for r in results]
    nses_array = np.array(nses)
    mean_nse = np.nanmean(nses_array)
    std_nse = np.nanstd(nses_array)

    print(f"Mean NSE: {mean_nse:.4f}")
    print(f"Std NSE:  {std_nse:.4f}")
    preds_all = np.stack(preds_all, axis=1)  
    trues_all = np.stack(trues_all, axis=1)
    train_time = np.sum(train_times)
    infer_time = np.sum(infer_times)

    param_names = list(next(iter(station_params.values())).keys())
    # print(preds_all.shape)
    # print(trues_all.shape)
    trues_all=trues_all
    preds_all=preds_all.squeeze()
    # print(preds_all.shape)
    # print(trues_all.shape)
    # nan_count_p = np.isnan(preds_all).sum()
    # nan_count_t = np.isnan(trues_all).sum()
    # print(nan_count_p )
    # print(nan_count_t)
    preds_all= np.nan_to_num(preds_all, nan=0.0)

    metrics = compute_avg_metrics(
        trues_all,
        preds_all
    )

    stats = {
        'train_time': train_time,
        'infer_time': infer_time,
        'num_params': len(param_names)
    }

    logger.info(
        f"Run {run_id}: train={train_time:.2f}s, infer={infer_time:.2f}s"
    )

    with open(os.path.join(run_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=4)

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=4)

    np.savez(
        os.path.join(run_dir, "predictions.npz"),
        predictions=preds_all,
        targets=trues_all,
    )
    
    return metrics, preds_all, stats

def aggregate_results(cfg, all_metrics, logger):
    """Aggregate metrics across all runs"""
    
    df_metrics = pd.DataFrame(all_metrics)
    
    summary = {
        'metric': [],
        'mean': [],
        'std': [],
        'min': [],
        'max': []
    }
    
    for col in df_metrics.columns:
        summary['metric'].append(col)
        summary['mean'].append(df_metrics[col].mean())
        summary['std'].append(df_metrics[col].std())
        summary['min'].append(df_metrics[col].min())
        summary['max'].append(df_metrics[col].max())
    
    df_summary = pd.DataFrame(summary)

    df_summary.to_csv(os.path.join(cfg['results_dir'], 'metrics_summary.csv'), index=False)
    df_metrics.to_csv(os.path.join(cfg['results_dir'], 'all_runs_metrics.csv'), index=False)

    logger.info("\nAggregated metrics:\n")
    logger.info(df_summary.to_string(index=False))
    logger.info(df_metrics.to_string(index=False))
    logger.info(f"\nSaved to {cfg['results_dir']}")

    return df_summary

def aggregate_stats(cfg, all_stats, logger):
    """Aggregate FLOPs/params/time stats across runs"""

    df_stats = pd.DataFrame(all_stats)

    summary = {
        "stat": [],
        "mean": [],
        "std": [],
        "min": [],
        "max": []
    }

    for col in df_stats.columns:
        summary["stat"].append(col)
        summary["mean"].append(df_stats[col].mean())
        summary["std"].append(df_stats[col].std())
        summary["min"].append(df_stats[col].min())
        summary["max"].append(df_stats[col].max())

    df_summary = pd.DataFrame(summary)
    
    df_summary.to_csv(os.path.join(cfg['results_dir'], 'stats_summary.csv'), index=False)
    df_stats.to_csv(os.path.join(cfg['results_dir'], 'all_runs_stats.csv'), index=False)

    logger.info("\nAggregated runtime & FLOPs stats:\n")
    logger.info(df_summary.to_string(index=False))
    logger.info(df_stats.to_string(index=False))
    logger.info(f"\nSaved to {cfg['results_dir']}")
    
    return df_summary

@hydra.main(version_base=None, config_path="../config", config_name="hbv.yaml")
def main(cfg: DictConfig):
    
    # Create directories
    os.makedirs(cfg['results_dir'], exist_ok=True)
    os.makedirs(cfg['log_dir'], exist_ok=True)
    logger = get_logger(os.path.join(cfg['log_dir'], "experiment.log"))
    logger.info(f"Loaded config: {cfg}")

    # Prepare data
    logger.info("\nPreparing data...")
    dt = DataPreparer(cfg)
    prepared = dt.load_prepared_data("data")

    clim_station_order = prepared["clim_station_order"]
    num_stations    = prepared["num_nodes"]
    dates        = prepared["dates"]
    
    # Save config copy
    with open(os.path.join(cfg['results_dir'], "config.yaml"), "w") as f:
        yaml.dump(cfg, f)
    
    temp_all = pd.read_csv("data/temp_aligned.csv")
    pet_all = pd.read_csv("data/pet_aligned.csv")
    prec_all = pd.read_csv("data/prec_aligned.csv")

    inputs = np.load("data/inputs_unwindowed.npy")
    targets = np.load("data/targets_unwindowed.npy")
    areas = pd.read_csv("data/catchment_area.csv")
    temp_all['date'] = pd.to_datetime(temp_all['date'])
    pet_all['date'] = pd.to_datetime(pet_all['date'])
    prec_all['date'] = pd.to_datetime(prec_all['date'])
    dates = pd.to_datetime(dates)
    
    # Filter and sort each dataframe by date
    ordered_columns = ["date"] + list(clim_station_order)
    temp_all = temp_all[ordered_columns] 

    temp_all = temp_all[temp_all['date'].isin(dates)].set_index('date').sort_index()

    areas["station_id"]=areas["station_id"].astype(str)
    areas['station_id'] = pd.Categorical(
        areas['station_id'], 
        categories=clim_station_order, 
        ordered=True
    )

    areas = areas.sort_values('station_id').dropna(subset=['station_id'])
    areas = np.array(areas)
    temp = np.array(temp_all)

    pet_all = pet_all[ordered_columns]   
    pet_all = pet_all[pet_all['date'].isin(dates)].set_index('date').sort_index()
    pet = np.array(pet_all)
    
    prec_all = prec_all[ordered_columns] 
    prec_all = prec_all[prec_all['date'].isin(dates)].set_index('date').sort_index()

    prec = np.array(prec_all)

    train_end = int(cfg['train_ratio'] * len(dates))
    val_end = train_end + int(cfg['val_ratio'] * len(dates))

    train_inputs= inputs[:train_end] 
    val_inputs  = inputs[train_end:val_end]
    test_inputs = inputs[val_end:]

    train_targets=targets[:train_end] 
    val_targets = targets[train_end:val_end]
    test_targets= targets[val_end:]

    train_dates = dates[:train_end] 
    val_dates  = dates[train_end:val_end]
    test_dates = dates[val_end:]

    train_pet = pet[:train_end] 
    val_pet  = pet[train_end:val_end]
    pet_test = pet[val_end:]
    
    train_temp = temp[:train_end] 
    val_temp  = temp[train_end:val_end]
    temp_test = temp[val_end:]

    train_prec = prec[:train_end] 
    val_prec  = prec[train_end:val_end]
    prec_test = prec[val_end:]

    temp_train = np.concatenate([train_temp, val_temp], axis=0)
    pet_train = np.concatenate([train_pet, val_pet], axis=0)
    prec_train = np.concatenate([train_prec, val_prec], axis=0)

    train_inputs = np.concatenate([train_inputs, val_inputs], axis=0)  
    train_targets = np.concatenate([train_targets, val_targets], axis=0)
    month_train = np.concatenate([train_dates, val_dates], axis=0)   
    month_train = (month_train.astype('datetime64[M]').astype(int) % 12) + 1
    month_test = ((np.array(test_dates)).astype('datetime64[M]').astype(int) % 12) + 1  
    pet_m_test = compute_monthly_means(pet_test, month_test)
    temp_m_test = compute_monthly_means(temp_test, month_test)
    
    pet_m = compute_monthly_means(pet_train, month_train)
    temp_m = compute_monthly_means(temp_train, month_train)
    prec = prec_train

    qobs = (train_targets[:, :] * 86.4)/ areas[np.newaxis,:,1]
    prec_test = prec_test
    qobs_test = (test_targets[:, :] * 86.4)/ areas[np.newaxis,:,1]
    # print(qobs[:,89])
    # print(qobs[0,1])
    # print(test_inputs[:, 89, 12])
    # print(test_inputs[0, 0, 12])
    # print(areas[np.newaxis, 0, 1])
    # print(f"Precip (mm/day): {prec[:5]}")        
    # print(f"Temp (C): {temp[:5]}")               
    # print(f"Q_obs (mm/day): {qobs[:5]}")
    
    # Run experiments
    all_metrics = []
    all_stats = [] 
    for run_id in range(cfg['num_runs']):
        metrics, _, run_stats = run_single_experiment(cfg, 
                        run_id, 
                        num_stations,
                        logger, 
                        prec_test,
                        temp_test,
                        month_test,
                        pet_m_test,
                        temp_m_test,
                        prec,
                        temp_train,
                        month_train,
                        pet_m,
                        temp_m,
                        qobs,
                        qobs_test
                        )
        all_metrics.append(metrics)
        all_stats.append(run_stats)
    
    # Aggregate results
    aggregate_results(cfg, all_metrics, logger)
    aggregate_stats(cfg, all_stats, logger)
    gc.collect()
    del all_metrics, all_stats
    logger.info("\nRuns completed")

if __name__ == '__main__':
    main()