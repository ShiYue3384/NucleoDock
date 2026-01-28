
import math
import warnings
from bisect import bisect_right
from typing import List

import torch
from torch.optim.lr_scheduler import _LRScheduler, CyclicLR

__all__ = ["WarmupMultiStepLR", "WarmupCosineAnnealingLR"]


class WarmupExponentialLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, gamma, last_epoch=-1, warmup_iters=2, warmup_factor=1.0 / 3, verbose=False, **kwargs):
        self.gamma = gamma
        self.warmup_method = 'linear'
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        warmup_factor = _get_warmup_factor_at_iter(self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor)

        if self.last_epoch <= self.warmup_iters:
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma**self.last_epoch for base_lr in self.base_lrs]


class WarmupLinearLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer,
                 max_iters=1000,
                 last_epoch=-1,
                 warmup_iters=2,
                 warmup_factor=1.0 / 3,
                 verbose=False,
                 offset=0,
                 **kwargs):
        self.warmup_method = 'linear'
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor
        self.max_iters = max_iters
        self.offset = offset
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        warmup_factor = _get_warmup_factor_at_iter(self.warmup_method,
                                                   self.last_epoch,
                                                   self.warmup_iters,
                                                   self.warmup_factor,
                                                   offset=self.offset)

        real_step = self.last_epoch + self.offset

        if real_step <= self.warmup_iters:
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        return [base_lr * self.lr_lambda() for base_lr in self.base_lrs]

    def lr_lambda(self):
        return max(0.0,
                   float(self.max_iters - (self.last_epoch + self.offset)) / float(max(1, self.max_iters - self.warmup_iters)))

    def update_offset(self, offset):
        self.offset = offset


def _get_warmup_factor_at_iter(method: str, iter: int, warmup_iters: int, warmup_factor: float, offset=0) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See https://arxiv.org/abs/1706.02677 for more details.
    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).
    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter + offset >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = (iter + offset) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))


class OffsetCyclicLR(CyclicLR):

    def __init__(self, *args, offset=0, **kwargs):
        self.offset = offset
        super().__init__(*args, **kwargs)

    def get_lr(self):
        """Calculates the learning rate at batch index. This function treats
        `self.last_epoch` as the last batch index.

        If `self.cycle_momentum` is ``True``, this function has a side effect of
        updating the optimizer's momentum.
        """

        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        cycle = math.floor(1 + (self.offset + self.last_epoch) / self.total_size)
        x = 1. + (self.offset + self.last_epoch) / self.total_size - cycle
        if x <= self.step_ratio:
            scale_factor = x / self.step_ratio
        else:
            scale_factor = (x - 1) / (self.step_ratio - 1)

        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            base_height = (max_lr - base_lr) * scale_factor
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.offset + self.last_epoch)
            lrs.append(lr)

        if self.cycle_momentum:
            momentums = []
            for base_momentum, max_momentum in zip(self.base_momentums, self.max_momentums):
                base_height = (max_momentum - base_momentum) * scale_factor
                if self.scale_mode == 'cycle':
                    momentum = max_momentum - base_height * self.scale_fn(cycle)
                else:
                    momentum = max_momentum - base_height * self.scale_fn(self.offset + self.last_epoch)
                momentums.append(momentum)
            for param_group, momentum in zip(self.optimizer.param_groups, momentums):
                param_group['momentum'] = momentum

        return lrs

    def update_offset(self, offset):
        self.offset = offset
