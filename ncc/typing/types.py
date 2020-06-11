# -*- coding: utf-8 -*-

from typing import (
    Optional,
    Sequence, List, Tuple,
    NoReturn,
    Any,
    Mapping, Dict,
    Set,
    Callable,
    Union,
    Iterator, Iterable
)
from torch import (
    Tensor,
    LongTensor,
    FloatTensor,
    BoolTensor,
)
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module

from collections import (
    OrderedDict, defaultdict,
    Counter,
)
from abc import ABC
from numba import (
    int8, uint8, int16, uint16, int32, uint32, int64, uint64,
    intc, uintc,  # c language int type
    float32, float64,
)

'''
Type aliases for NCC libraries:
'''
# define function
Any_t = Any
Const_t = Any
Void_t = Optional[NoReturn]
Func_t = Callable
Class_t = Union[object, ABC]
String_t = str
Int_t = Union[
    intc, uintc,
    int8, uint8, int16, uint16, int32, uint32, int64, uint64, int,
]
Float_t = Union[
    float, float32, float64,
]
Number_t = Union[Int_t, Float_t]
Bool_t = bool
Byte_t = bytes
Exception_t = Exception
Sequence_t = Union[Sequence, List, Tuple]  # for list & tuple
Dict_t = Union[Mapping, Dict, OrderedDict, defaultdict]
Set_t = Union[Set, frozenset]
Iterator_t = Union[Iterator, Iterable]

# torch Tensor
Tensor_t = Optional[Tensor]
LTensor_t = LongTensor
FTensor_t = FloatTensor
BTensor_t = Optional[BoolTensor, FloatTensor]  # for mask
ThOptimizer_t = Optimizer
ThDataset_t = Dataset
ThDataLoader_t = DataLoader
ThNetwork_t = Module

# other tools
Counter_t = Counter

__all__ = (
    'String_t', 'Int_t', 'Float_t', 'Number_t', 'Bool_t', 'Byte_t',
    'Any_t', 'Const_t', 'Void_t', 'Func_t', 'Class_t', 'Exception_t',
    'Sequence_t', 'Dict_t', 'Set_t', 'Iterator_t',

    'Tensor_t', 'LTensor_t', 'FTensor_t', 'BTensor_t',
    'ThDataset_t', 'ThDataLoader_t',
    'ThNetwork_t', 'ThOptimizer_t',

    'Counter_t',
)
