"""
@Time   :   2021-01-21 10:51:21
@File   :   build.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import torch
from src.solver import lr_scheduler
from src.solver.lr_scheduler import OffsetCyclicLR
from lion_pytorch import Lion


def make_optimizer(cfg, model, **kwargs):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"name": key, "params": [value], "lr": lr, "weight_decay": weight_decay, "freeze": False}]
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM, **kwargs)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'Lion':
        optimizer = Lion(params, **kwargs, betas=(0.95, 0.98))
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, **kwargs)

    return optimizer


def build_lr_scheduler(cfg, optimizer, **kwargs):
    scheduler_args = {
        "optimizer": optimizer,

        # warmup options
        "warmup_factor": cfg.SOLVER.WARMUP_FACTOR,
        "warmup_iters": int(cfg.SOLVER.MAX_STEPS * cfg.SOLVER.WARMUP_STEP_RATIO),
        "warmup_method": cfg.SOLVER.WARMUP_METHOD,

        # multi-step lr scheduler options
        "gamma": cfg.SOLVER.GAMMA,

        # cosine annealing lr scheduler options
        "max_iters": cfg.SOLVER.MAX_STEPS,
    }
    if cfg.SOLVER.SCHED == 'CyclicLR':
        offset = kwargs.pop('offset')
        if offset is None:
            offset = 0
        scheduler = OffsetCyclicLR(optimizer,
                                   base_lr=cfg.SOLVER.BASE_LR * 0.05,
                                   max_lr=cfg.SOLVER.BASE_LR,
                                   step_size_up=int(cfg.SOLVER.MAX_STEPS * cfg.SOLVER.WARMUP_STEP_RATIO),
                                   cycle_momentum=False,
                                   offset=offset)
    else:
        scheduler = getattr(lr_scheduler, cfg.SOLVER.SCHED)(**scheduler_args, **kwargs)
    return scheduler
