import math
from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler

class WarmupMultiStepLR(MultiStepLR):
    r"""
    # max_iter = epochs * steps_per_epoch
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_iter (int): The total number of steps.
        milestones (list) – List of iter indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay. Default: 0.1.
        pct_start (float): The percentage of the cycle (in number of steps) spent
                    increasing the learning rate.
                    Default: 0.3
        warmup_factor (float):         
        last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(self, optimizer, max_iter, milestones, gamma=0.1, pct_start=0.3, warmup_factor=1.0 / 2,
                  last_epoch=-1):
        self.warmup_factor = warmup_factor
        self.warmup_iters = int(pct_start * max_iter)
        super().__init__(optimizer, milestones, gamma, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.warmup_iters:
            alpha = self.last_epoch / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [lr * warmup_factor for lr in self.base_lrs]
        else:
            lr = super().get_lr()
        return lr

class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer, max_iter, pct_start=0.3, warmup_factor=1.0 / 3, 
                 eta_min=0, last_epoch=-1):
        self.warmup_factor = warmup_factor
        self.warmup_iters = int(pct_start * max_iter)
        self.max_iter, self.eta_min = max_iter, eta_min
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch <= self.warmup_iters:
            alpha = self.last_epoch / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [lr * warmup_factor for lr in self.base_lrs]
        else:
            # print ("after warmup")
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(
                        math.pi * (self.last_epoch - self.warmup_iters) / (self.max_iter - self.warmup_iters))) / 2
                    for base_lr in self.base_lrs]

class WarmupPolyLR(_LRScheduler):
    def __init__(self, optimizer, T_max, pct_start=0.3, warmup_factor=1.0 / 4, 
                 eta_min=0, power=0.9):
        self.warmup_factor = warmup_factor
        self.warmup_iters = int(pct_start * T_max)
        self.power = power
        self.T_max, self.eta_min = T_max, eta_min
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch <= self.warmup_iters:
            alpha = self.last_epoch / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [lr * warmup_factor for lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    math.pow(1 - (self.last_epoch - self.warmup_iters) / (self.T_max - self.warmup_iters),
                             self.power) for base_lr in self.base_lrs]
