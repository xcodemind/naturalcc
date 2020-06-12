# -*- coding: utf-8 -*-

from typing import (
    Optional, Union
)

from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module

# torch Tensor
# torch version conflict
try:
    from torch import (
        Tensor,
        LongTensor,
        FloatTensor,
        BoolTensor,
    )

    Tensor_t = Optional[Tensor]
    LTensor_t = LongTensor
    FTensor_t = FloatTensor
    BTensor_t = Union[BoolTensor, FloatTensor, None]  # for mask
except:
    from torch import Tensor

    Tensor_t = Optional[Tensor]
    LTensor_t = Tensor
    FTensor_t = Tensor
    BTensor_t = Optional[Tensor]  # for mask

ThOptimizer_t = Optimizer
ThDataset_t = Dataset
ThDataLoader_t = DataLoader
ThNetwork_t = Module

__all__ = (
    'Tensor_t', 'LTensor_t', 'FTensor_t', 'BTensor_t',
    'ThDataset_t', 'ThDataLoader_t',
    'ThNetwork_t', 'ThOptimizer_t',
)
