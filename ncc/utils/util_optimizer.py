# -*- coding: utf-8 -*-
from bisect import bisect
from torch.optim.optimizer import Optimizer
from torch.optim import lr_scheduler
from ncc import *

def create_scheduler(optimizer: Optimizer, warmup_epochs: int, warmup_factor:float, lr_milestones:List, lr_gamma: float):

    def _lr_lambda_fun(current_epoch: int) -> float:
        """Returns a learning rate multiplier.
        Till `warmup_epochs`, learning rate linearly increases to `initial_lr`,
        and then gets multiplied by `lr_gamma` every time a milestone is crossed.
        """
        # current_epoch = float(current_iteration) / iterations
        if current_epoch <= warmup_epochs:
            alpha = current_epoch / float(warmup_epochs)
            return warmup_factor * (1.0 - alpha) + alpha
        else:
            idx = bisect(lr_milestones, current_epoch)
            return pow(lr_gamma, idx)

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda_fun)

    return scheduler
