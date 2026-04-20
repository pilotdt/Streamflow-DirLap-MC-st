"""
Example usage: python3 src/main_arma.py 
"""

import hydra
import os
from omegaconf import DictConfig, OmegaConf
OmegaConf.register_new_resolver("basename", lambda x: os.path.basename(x).split(".")[0])
import yaml
import json
import numpy as np
import pandas as pd
import time 

from utils.seed import set_seed
from utils.logger import get_logger
from evaluation.compute_metrics import compute_avg_metrics

from data.dataset import (
    load_flow_csv,
    load_other_csv
)
from data.dataloader import DataPreparer
from models.arma import ARMA


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_data(cfg, specific_order):
    """Load and prepare data for ARMA"""
    if cfg["other_csv"]=='':
        dates, flow_df = load_flow_csv(cfg["flow_csv"])
        
        flow_df = flow_df.drop(columns="Unnamed: 0", errors="ignore")
        flow_df.columns = flow_df.columns.astype(str)
        flow_df = flow_df[specific_order]
        # Endogenous variables
        y = flow_df.values  
    else:
        dates, other_df = load_other_csv(cfg['other_csv'])    
        X_full = other_df.values
        y = X_full

    n = len(y)
    train_end = int(cfg["train_ratio"] * n)

    y_train = y[:train_end]
    y_test = y[train_end:]

    H = cfg["horizon"]
    B, N = y_test.shape
    B_full = B - H + 1
    trues = np.stack([
        y_test[b : b + H]
        for b in range(B_full)
    ]) 

    return y_train, y_test, trues

def run_single_experiment(cfg, run_id, y_train, y_test, trues, logger):
    """Run a single experiment with a specific seed"""

    seed = cfg["start_seed"] + run_id
    set_seed(seed)

    run_dir = os.path.join(cfg["results_dir"], f"run_{run_id + 1}")
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"\n{'='*60}")
    logger.info(f"Run {run_id+1}/{cfg['num_runs']} - Seed: {seed}")
    logger.info(f"{'='*60}")

    start_train = time.time()

    preds, metrics_internal = ARMA(
        y_train=y_train,
        y_test=y_test,
        horizon=cfg["horizon"],
        order=(cfg.get("p",1), cfg.get("d",0), cfg.get("q",0)), 
    )

    end_train = time.time()
    train_time = end_train - start_train


    metrics = compute_avg_metrics(
        trues=trues,
        preds=preds,
    )

    run_stats = {
        "train_time_total": train_time,
        "train_time_per_run": train_time,
    }

    with open(os.path.join(run_dir, "stats.json"), "w") as f:
        json.dump(run_stats, f, indent=4)

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=4)

    np.savez(
        os.path.join(run_dir, "predictions.npz"),
        predictions=preds,
        targets=trues,
    )

    logger.info("Run metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    return metrics, run_stats


def aggregate_results(cfg, all_metrics, logger):
    df = pd.DataFrame(all_metrics)

    summary = {
        "metric": df.columns,
        "mean": df.mean().values,
        "std": df.std().values,
        "min": df.min().values,
        "max": df.max().values,
    }

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(os.path.join(cfg["results_dir"], "metrics_summary.csv"), index=False)
    df.to_csv(os.path.join(cfg["results_dir"], "all_runs_metrics.csv"), index=False)

    logger.info("\nAggregated metrics:\n")
    logger.info(df_summary.to_string(index=False))


def aggregate_stats(cfg, all_stats, logger):
    df = pd.DataFrame(all_stats)
    df.to_csv(os.path.join(cfg["results_dir"], "all_runs_stats.csv"), index=False)


@hydra.main(version_base=None, config_path="../config", config_name="arma")
def main(cfg: DictConfig):

    os.makedirs(cfg["results_dir"], exist_ok=True)
    os.makedirs(cfg["log_dir"], exist_ok=True)

    logger = get_logger(os.path.join(cfg["log_dir"], "experiment.log"))
    logger.info(f"Loaded config: {cfg}")

    logger.info("Preparing data...")
    dt = DataPreparer(cfg)
    prepared = dt.load_prepared_data("data")
    if cfg["other_csv"]=="":
        clim_station_order = prepared["clim_station_order"]
    else:
        clim_station_order = None


    y_train, y_test, trues = prepare_data(cfg, specific_order=clim_station_order)

    with open(os.path.join(cfg["results_dir"], "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    all_metrics = []
    all_stats = []

    for run_id in range(cfg["num_runs"]):
        metrics, stats = run_single_experiment(
            cfg,
            run_id,
            y_train,
            y_test,
            trues,
            logger,
        )
        all_metrics.append(metrics)
        all_stats.append(stats)

    aggregate_results(cfg, all_metrics, logger)
    aggregate_stats(cfg, all_stats, logger)
    logger.info("\nRuns completed.")


if __name__ == "__main__":
    main()
