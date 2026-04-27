"""
Optimisation for sLSTM
Usage: python3 src/optimise_prms_slstm.py reg_4_loss=""
Or:    python3 src/optimise_prms_slstm.py reg_4_loss="L_dir"
Or:    python3 src/optimise_prms_slstm.py reg_4_loss="L_dir" add_storage=True
"""

import hydra
from omegaconf import DictConfig
import optuna
import torch
import yaml
import copy
import os
import gc
import time

from data.dataloader import DataPreparer
from data.adjacency import load_adjacency_matrix
from data.dataset import RiverFlowDataset
from torch.utils.data import DataLoader
from utils.seed import set_seed, seed_worker
from utils.logger import get_logger
from training.optimizer_factory import create_optimizer
from training.utils import build_advection_operator
from models.slstm import sLSTM
from training.losses import  MSELoss
from training.trainer import Trainer
from evaluation.compute_metrics import compute_avg_metrics_fast
from evaluation.evaluator import Evaluator

def run_single_experiment(cfg, run_id, train_loader, val_loader, std_per_station, seq_lengths,
                          test_loader, clim_station_order, num_stations, in_features, y_scaler, device, logger, trial=None):
    """Run a single experiment with a specific seed"""
    
    # Set seed
    seed = cfg['start_seed'] + run_id
    set_seed(seed)
    
    ckpt_path = None
        
    # Create criterion
    if cfg["reg_4_loss"] == "L_dir":
        A_np = cfg['adj_np']    
        A_np, _, _ = load_adjacency_matrix(A_np, specific_order=clim_station_order)
        A = torch.tensor(A_np, dtype=torch.float32, device=device)
        if cfg["add_storage"]==False:
            L_dir = build_advection_operator(A)
        elif cfg["add_storage"]:
            L_dir = None
    elif cfg["reg_4_loss"] == "Nl_phys_term":
        A_np = cfg['adj_np']    
        A_np, _, _ = load_adjacency_matrix(A_np, specific_order=clim_station_order)
        A = torch.tensor(A_np, dtype=torch.float32, device=device)
        L_dir = build_advection_operator(A)
    else:
        L_dir = None
            
    # Initialize model
    if cfg["reg_4_loss"] == "L_dir" and cfg["add_storage"]:
        model = sLSTM(
            num_stations=num_stations,
            in_features=in_features,
            hidden=cfg['hidden'],
            add_storage=cfg["add_storage"],
            A=A,
            horizon=cfg['horizon'],
            batch_size=cfg["batch_size"],
            slstm_kernel_size=cfg.get('xlstm_kernel_size'),
            slstm_num_heads=cfg.get('xlstm_heads'),
            slstm_num_blocks=cfg.get('xlstm_num_blocks'),
            slstm_proj_factor=cfg.get('xlstm_proj_factor'),
            dropout=cfg.get('dropout', 0.1)
        ).to(device)
    elif cfg["reg_4_loss"] == "Nl_phys_term":
        model = LSTM(
            num_stations=num_stations,
            in_features=in_features,
            hidden=cfg['hidden'],
            horizon=cfg['horizon'],
            batch_size=cfg["batch_size"],
            slstm_kernel_size=cfg.get('xlstm_kernel_size'),
            slstm_num_heads=cfg.get('xlstm_heads'),
            slstm_num_blocks=cfg.get('xlstm_num_blocks'),
            slstm_proj_factor=cfg.get('xlstm_proj_factor'),
            dropout=cfg.get('dropout')
            lambda_nl_reg=cfg["lambda_nl_reg"],
            ).to(device) 
    else:
        model = sLSTM(
            num_stations=num_stations,
            in_features=in_features,
            hidden=cfg['hidden'],
            horizon=cfg['horizon'],
            batch_size=cfg["batch_size"],
            slstm_kernel_size=cfg.get('xlstm_kernel_size'),
            slstm_num_heads=cfg.get('xlstm_heads'),
            slstm_num_blocks=cfg.get('xlstm_num_blocks'),
            slstm_proj_factor=cfg.get('xlstm_proj_factor'),
            dropout=cfg.get('dropout')
        ).to(device)
    # Create optimizer config
    optimizer_cfg = {
        'optimizer': cfg['optimizer'],
        'lr': cfg['lr'],
        'weight_decay': cfg['weight_decay']
    }
    
    # Create optimizer
    optimizer = create_optimizer(model, optimizer_cfg)

    criterion = MSELoss()
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        logger=logger,
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
        add_storage=cfg["add_storage"],
        L_dir=L_dir if cfg["reg_4_loss"] == "L_dir" and cfg["add_storage"] else None,
        lambda_L_dir=cfg["lambda_L_dir"] if cfg["reg_4_loss"] == "L_dir" else None,
        lambda_nl_reg = cfg["lambda_nl_reg"] if cfg['reg_4_loss'] == "Nl_phys_term" else None,
        epochs=cfg['epochs']
    )
    end_train = time.time()
    train_time_total = end_train - start_train
    logger.info(f"Training time: {train_time_total:.2f} sec")
    
    evaluator = Evaluator(model, device)

    # Evaluate model 
    start_inf = time.time()
    preds_scaled, trues_scaled = evaluator.evaluate(val_loader, add_storage=cfg["add_storage"],lambda_nl_reg=cfg["lambda_nl_reg"])
    end_inf = time.time()
    inf_total = end_inf - start_inf
    logger.info(f"Inference time total: {inf_total:.4f} sec")
    preds = y_scaler.inverse_transform(preds_scaled)
    trues = y_scaler.inverse_transform(trues_scaled)
    preds = preds.squeeze(-1)
    trues = trues.squeeze(-1)
    metrics = compute_avg_metrics_fast(trues, preds)

    gc.collect()
    torch.cuda.empty_cache()
    return metrics, preds, trues

def objective(trial, base_cfg, train_dataset, val_dataset, prepared, device, logger):

    cfg = copy.deepcopy(base_cfg)

    # cfg["lr"] = trial.suggest_float("lr", 1e-5, 3e-3, log=True)

    # if cfg["optimizer"] == "adam":
    #    cfg["weight_decay"] = trial.suggest_float(
    #        "weight_decay", 1e-5, 1e-2, log=True
    #    )
    # else:
    #    cfg["weight_decay"] = 0.0

    # # if cfg["optimizer"] == "rmsprop":
    # #     cfg["alpha"] = trial.suggest_float("alpha", 0.9, 0.99)
    # #     cfg["momentum"] = trial.suggest_float("momentum", 0.0, 0.9)
    
    # cfg["batch_size"] = trial.suggest_categorical("batch_size", [8, 16, 32])

    # cfg["hidden"] = trial.suggest_categorical("hidden", [32, 64, 128])
    # # cfg['xlstm_num_blocks'] = trial.suggest_int('xlstm_num_blocks', 1, 3, 4)
    # # cfg['xlstm_kernel_size'] = trial.suggest_int('xlstm_kernel_size', 1, 3, 7)
    # # cfg['xlstm_heads'] = trial.suggest_categorical('xlstm_heads', [1, 2, 4])
    # cfg['xlstm_proj_factor'] = trial.suggest_categorical('xlstm_proj_factor', [1, 1.5, 2.0])
    # cfg['dropout'] = trial.suggest_categorical('dropout', [0.0, 0.1, 0.2])

    if cfg["reg_4_loss"]=="L_dir":
        # cfg["lambda_L_dir"] = trial.suggest_float("lambda_L_dir", 1e-5, 1e2, log=True)   
        # cfg["lambda_L_dir"] = trial.suggest_float("lambda_L_dir", 1e-1, 1, log=True)    
        cfg["lambda_L_dir"] = trial.suggest_categorical("lambda_L_dir", [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])  

    logger.info(f"Trial {trial.number} hyperparameters: {trial.params}")

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2, worker_init_fn=seed_worker, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=2, pin_memory=True, worker_init_fn=seed_worker, persistent_workers=True, prefetch_factor=2, drop_last=True)
    
    try:
        metrics, _, _ = run_single_experiment(
            cfg=cfg,
            run_id=0,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=None,            
            clim_station_order=prepared["clim_station_order"],
            num_stations=prepared["num_nodes"],
            in_features=prepared["in_features"],
            seq_lengths=prepared["seq_lengths"],
            std_per_station=prepared["std_per_station"],
            y_scaler=prepared["y_scaler"],
            device=device,
            logger=logger,
            trial=None
        )
        mae = metrics["MAE"]
        rmse = metrics["RMSE"]
        
        del metrics

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return mae, rmse
    except RuntimeError as e:
        logger.info(e)
        print("Error encountered, pruning trial.")
        raise optuna.TrialPruned()

@hydra.main(version_base=None, config_path="../config", config_name="slstm.yaml")
def main(cfg: DictConfig):
    os.makedirs(base_cfg["log_dir"], exist_ok=True)
    logger = get_logger(os.path.join(base_cfg["log_dir"], "optuna_val.log"))

    device = torch.device(
        "cuda" if base_cfg["device"] == "cuda" and torch.cuda.is_available() else "cpu"
    )
    
    obj = DataPreparer(base_cfg)
    prepared = obj.load_prepared_data("data")
    
    train_dataset = RiverFlowDataset(
        prepared["X_train"][...,:],
        prepared["y_train"].squeeze(-1) 
    )
    val_dataset = RiverFlowDataset(
        prepared["X_val"][...,:],
        prepared["y_val"].squeeze(-1) 
    )

    clim_station_order = prepared["clim_station_order"]
    num_stations    = prepared["num_nodes"]
    in_features  = prepared["in_features"]

    dates        = prepared["dates"]
    y_scaler     = prepared["y_scaler"]
    seq_lengths  = prepared["seq_lengths"]

    # Set device and multi-GPU
    if base_cfg['device'] == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(base_cfg['device'])

    study = optuna.create_study(
        directions=["minimize", "minimize"]  # MAE, RMSE
    )

    study.optimize(
        lambda trial: objective(trial, base_cfg, train_dataset, val_dataset, prepared, device, logger),
        n_trials=cfg["trials"]
    )

    logger.info("Optimal solutions:")
    for t in study.best_trials:
        logger.info(
            f" MAE={t.values[0]:.4f}, RMSE={t.values[1]:.4f}, params={t.params}"
        )

    best_trial = min(study.best_trials, key=lambda t: t.values[0])

    best_cfg = copy.deepcopy(base_cfg)
    best_cfg.update(best_trial.params)

    with open(os.path.join(base_cfg["results_dir"], "best_config.yaml"), "w") as f:
        yaml.dump(best_cfg, f)

    logger.info("Best validation config saved")


if __name__ == "__main__":
    main()
