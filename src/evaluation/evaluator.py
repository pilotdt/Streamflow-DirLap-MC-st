import numpy as np
import torch


class Evaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def evaluate(self, dataloader, add_storage=False, lambda_nl_reg=None):
        self.model.eval()
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(self.device)
                y_true = y.numpy()
                if add_storage:
                    pred, _ = self.model(X)
                    pred = pred.cpu().numpy()
                elif lambda_nl_reg is not None:
                    pred, _, _ = self.model(X)
                    pred = pred.cpu().numpy()
                else:
                    pred = self.model(X).cpu().numpy()
                all_preds.append(pred)
                all_trues.append(y_true)
        preds = np.concatenate(all_preds, axis=0)
        trues = np.concatenate(all_trues, axis=0)
        return preds, trues