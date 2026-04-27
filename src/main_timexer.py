"""
Example usage: python3 src/main_timexer.py
"""

import hydra
from omegaconf import DictConfig
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
import yaml
import json
import numpy as np
import pandas as pd
import torch
import time
from thop import profile

from data.dataloader import DataPreparer
from data.dataset import RiverFlowDataset
from torch.utils.data import DataLoader
from utils.seed import set_seed, seed_worker
from utils.logger import get_logger
from models.timexer import TimeXer
from training.losses import  MSELoss
from training.trainer import Trainer
from training.optimizer_factory import create_optimizer
from evaluation.evaluator import Evaluator
from evaluation.compute_metrics import compute_avg_metrics


def run_single_experiment(cfg, run_id, train_loader, val_loader, std_per_station, seq_lengths,
                          test_loader, clim_station_order, num_stations, in_features, y_scaler, device, logger):
    """Run a single experiment with a specific seed"""
    
    # Set seed
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    seed = cfg['start_seed'] + run_id
    set_seed(seed)
    
    # Create run directory
    run_dir = os.path.join(cfg['results_dir'], f'run_{run_id+1}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Create checkpoint path
    ckpt_path = os.path.join(run_dir, 'best_model.pt')

    # Initialize model
    model = TimeXer(
        enc_in=num_stations * in_features,
        in_features = in_features, 
        seq_len=cfg["history"],
        pred_len=cfg["horizon"],
        features = cfg["features"],
        task_name = cfg["task_name"],
        use_norm = cfg["use_norm"],
        dropout = cfg["dropout"],
        patch_len=cfg["patch_len"],            
        d_model=cfg["d_model"],
        d_ff=cfg["d_ff"],
        n_heads=cfg["n_heads"],
        e_layers = cfg["e_layers"],
        embed = cfg["embed"],
        freq=cfg["freq"],
        factor=cfg["factor"],
        activation=cfg["activation"]
        ).to(device)
        
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {num_params:,}")
    dummy_input = torch.randn(cfg["batch_size"], cfg['history'], num_stations* in_features).to(device)
    macs, params = profile(model, inputs=(dummy_input,))
    logger.info(f"FLOPs (MACs): {macs:,}")
    
    # Create optimizer config
    optimizer_cfg = {
        'optimizer': cfg['optimizer'],
        'lr': cfg['lr'],
        'weight_decay': cfg['weight_decay']
    }
    
    # Create optimizer
    optimizer = create_optimizer(model, optimizer_cfg)
    
    # Create criterion
    criterion = MSELoss()
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        logger=logger,
        ckpt_path=ckpt_path,
        patience=cfg['patience']
    )
    
    # Train model
    logger.info(f"\n{'='*60}")
    logger.info(f"Run {run_id+1}/{cfg['num_runs']} - Seed: {seed}")
    logger.info(f"{'='*60}")
    
    start_train = time.time()
    model = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        std_per_station=std_per_station,
        add_storage=False,
        L_dir=None,
        lambda_L_dir=None,
        epochs=cfg['epochs']
    )
    end_train = time.time()
    train_time_total = end_train - start_train
    logger.info(f"Training time: {train_time_total:.2f} sec")
    logger.info(f"Training time per epoch: {train_time_total / cfg['epochs']:.2f} sec")
    
    evaluator = Evaluator(model, device)

    # Evaluate model
    start_inf = time.time()
    preds_scaled, trues_scaled = evaluator.evaluate(test_loader, add_storage=cfg['add_storage'])
    end_inf = time.time()
    inf_total = end_inf - start_inf
    logger.info(f"Inference time total: {inf_total:.4f} sec")
    logger.info(f"Inference per batch: {inf_total / len(test_loader):.4f} sec")
    logger.info(f"Inference per sample: {inf_total / len(test_loader.dataset):.6f} sec")
    print(trues_scaled.shape)
    print(preds_scaled.shape)
    preds = y_scaler.inverse_transform(preds_scaled)
    trues = y_scaler.inverse_transform(trues_scaled)
    print(trues.shape)
    print(preds.shape)
    preds = preds.squeeze(-1)
    trues = trues.squeeze(-1)
    metrics = compute_avg_metrics(trues, preds)
    
    # Collect run stats
    run_stats = {
        "num_params": num_params,
        "macs": macs,
        "train_time_total": train_time_total,
        "train_time_per_epoch": train_time_total / cfg['epochs'],
        "inference_total": inf_total,
        "inference_per_batch": inf_total / len(test_loader),
        "inference_per_sample": inf_total / len(test_loader.dataset)
    }

    # Save stats
    stats_path = os.path.join(run_dir, 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump({k: float(v) for k, v in run_stats.items()}, f, indent=4)
   
    # Save metrics
    metrics_path = os.path.join(run_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=4)
    
    # Save predictions
    preds_path = os.path.join(run_dir, 'predictions.npz')
    np.savez(preds_path, predictions=preds, targets=trues)
    
    logger.info(f"\nRun {run_id+1} Results:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    gc.collect()
    if next(model.parameters()).is_cuda:
        torch.cuda.empty_cache()
    return metrics, preds, trues, run_stats


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

@hydra.main(version_base=None, config_path="../config", config_name="timexer.yaml")
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

    train_dataset = RiverFlowDataset(
        prepared["X_train"].reshape(prepared["X_train"].shape[0],prepared["X_train"].shape[1],-1),
        prepared["y_train"].squeeze(-1) 
    )
    val_dataset = RiverFlowDataset(
        prepared["X_val"].reshape(prepared["X_val"].shape[0],prepared["X_val"].shape[1],-1),
        prepared["y_val"].squeeze(-1) 
    )
    test_dataset = RiverFlowDataset(
        prepared["X_test"].reshape(prepared["X_test"].shape[0],prepared["X_test"].shape[1],-1),
        prepared["y_test"].squeeze(-1) 
    )

    clim_station_order = prepared["clim_station_order"]
    num_stations    = prepared["num_nodes"]
    in_features  = prepared["in_features"]
    dates        = prepared["dates"]
    y_scaler     = prepared["y_scaler"]
    seq_lengths  = prepared["seq_lengths"]
    std_per_station = prepared["std_per_station"]

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=2, pin_memory=True, worker_init_fn=seed_worker, persistent_workers=True, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=2, pin_memory=True, worker_init_fn=seed_worker, persistent_workers=True, prefetch_factor=2)
    
    logger.info(f"Data prepared: {num_stations} stations, {in_features} features")
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Set device and multi-GPU
    if cfg['device'] == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(cfg['device'])

    # Save config copy
    with open(os.path.join(cfg['results_dir'], "config.yaml"), "w") as f:
        yaml.dump(cfg, f)
    
    # Run experiments
    all_metrics = []
    all_stats = [] 
    for run_id in range(cfg['num_runs']):
        metrics, _, _, run_stats = run_single_experiment(
            cfg=cfg,
            run_id=run_id,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            clim_station_order=clim_station_order,
            num_stations=num_stations,
            in_features=in_features,
            seq_lengths = seq_lengths,
            std_per_station=std_per_station,
            y_scaler=y_scaler,
            device=device,
            logger=logger
        )    
        all_metrics.append(metrics)
        all_stats.append(run_stats)
    
    # Aggregate results
    aggregate_results(cfg, all_metrics, logger)
    aggregate_stats(cfg, all_stats, logger)
    del metrics, run_stats
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    logger.info("\nRuns completed")

if __name__ == '__main__':
    main()