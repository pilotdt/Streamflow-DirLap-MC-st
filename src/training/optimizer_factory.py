import torch


def create_optimizer(model, cfg):
    """
    cfg: dictionary containing 'lr', 'weight_decay', 'optimizer'
    """
    opt_name = cfg.get('optimizer', 'Adam').lower()
    lr = cfg.get('lr')
    weight_decay = cfg.get('weight_decay')

    if opt_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    elif opt_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=cfg["weight_decay"]
        )
    else:
        raise ValueError(f"Unknown optimizer {opt_name}")
