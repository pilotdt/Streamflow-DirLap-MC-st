import torch
import optuna
from utils.logger import get_logger
from .utils import dir_laplacian_regularizer, nl_physics_regularizer


class Trainer:

    def __init__(self, model, optimizer, criterion, device, logger=None, patience=10, ckpt_path=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = logger or get_logger()
        self.ckpt_path = ckpt_path
        self.patience = patience

    def train(self, train_loader, val_loader, std_per_station, add_storage, L_dir, lambda_L_dir,  epochs=100, trial=None):
        best_val = float('inf')
        best_epoch = 0

        for epoch in range(1, epochs+1):
            self.model.train()
            running_loss = 0
            train_reg = 0
            count = 0
            # with autocast():
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                if add_storage:
                    pred, L_dir = self.model(X)
                    pred = pred.to(self.device)
                    L_dir = L_dir.to(self.device)
                elif lambda_nl_reg is not None:
                        pred, a, b = self.model(X)
                        pred = pred.to(self.device)
                        a = a.to(self.device)
                        b = b.to(self.device)
                else:
                    pred = self.model(X).to(self.device)
                # loss_mse = self.criterion(pred, y, std_per_station)
                loss_mse = self.criterion(pred, y)
                if L_dir is None:
                    loss = loss_mse
                elif L_dir is not None:
                    if lambda_nl_reg is None and lambda_L_dir is not None:
                        loss = loss_mse + lambda_L_dir * dir_laplacian_regularizer(pred, L_dir)
                        reg = lambda_L_dir * dir_laplacian_regularizer(pred, L_dir)
                    elif lambda_nl_reg is not None and lambda_L_dir is None:
                        loss = loss_mse + lambda_nl_reg * nl_phys_regularizer(pred, L_dir, a, b)
                        reg = lambda_nl_reg * nl_phys_regularizer(pred, L_dir, a, b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                running_loss += loss.item() * X.size(0)
                if L_dir is not None:
                    train_reg += reg.item() * X.size(0)
                count += X.size(0)
            train_loss = running_loss / max(count,1)
            if L_dir is not None:
                train_reg = train_reg / max(count,1)

            # --- Validation ---
            self.model.eval()
            val_loss = 0
            val_reg = 0
            vcount = 0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    if add_storage:
                            pred, L_dir = self.model(X)
                            pred = pred.to(self.device)
                            L_dir = L_dir.to(self.device)
                    elif lambda_nl_reg is not None:
                        pred, a, b = self.model(X)
                        pred = pred.to(self.device)
                        a = a.to(self.device)
                        b = b.to(self.device)
                    else:
                        pred = self.model(X).to(self.device)
                    # loss_mse = self.criterion(pred, y, std_per_station)
                    loss_mse = self.criterion(pred, y)
                    if L_dir is None:
                        loss = loss_mse
                    elif L_dir is not None:
                        if lambda_nl_reg is None and lambda_L_dir is not None:
                            loss = loss_mse + lambda_L_dir * dir_laplacian_regularizer(pred, L_dir)
                            reg = lambda_L_dir * dir_laplacian_regularizer(pred, L_dir)
                        elif lambda_nl_reg is not None and lambda_L_dir is None:
                            loss = loss_mse + lambda_nl_reg * nl_phys_regularizer(pred, L_dir, a, b)
                            reg = lambda_nl_reg * nl_phys_regularizer(pred, L_dir, a, b)
                    val_loss += loss.item() * X.size(0)
                    if L_dir is not None:
                        val_reg += reg.item()  * X.size(0)
                    vcount += X.size(0)
            val_loss = val_loss / max(vcount,1)
            if L_dir is not None:
                val_reg = val_reg / max(vcount,1)

            reg_str = f" train_reg_term={train_reg:.5f}, val_reg_term={val_reg:.5f}," if (L_dir is not None) else ""
            
            if trial is not None:
                trial.report(val_loss, epoch)
                if trial.should_prune() or torch.isnan(torch.tensor(val_loss)):
                                    raise optuna.exceptions.TrialPruned()
            self.logger.info(f"Epoch {epoch:03d} train_loss={train_loss:.5f}, val_loss={val_loss:.5f},{reg_str}")           
            # Early Stopping
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_epoch = epoch
                if self.ckpt_path is not None:
                    torch.save({'model_state': self.model.state_dict(),
                                'epoch': epoch, 'val_loss': val_loss},
                            self.ckpt_path)
            elif epoch - best_epoch >= self.patience:
                self.logger.info(f"Early stopping at epoch {epoch}. Best epoch {best_epoch}")
                break
        
        if self.ckpt_path is not None:
            try:
                ckpt = torch.load(self.ckpt_path, map_location=self.device)
                self.model.load_state_dict(ckpt['model_state'])
            except Exception:
                pass

        return self.model


