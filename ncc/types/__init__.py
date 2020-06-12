# -*- coding: utf-8 -*-

'''
Type aliases for NCC libraries:
'''

from ._base import *
from ._torch import *
from ._counter import *

__all__ = (
    'String_t', 'Int_t', 'Float_t', 'Number_t', 'Bool_t', 'Byte_t',
    'Any_t', 'Const_t', 'Void_t', 'Func_t', 'Class_t', 'Exception_t',
    'Sequence_t', 'Dict_t', 'Set_t', 'Iterator_t',

    'Optional', 'Union', 'Tuple',

    'Tensor_t', 'LTensor_t', 'FTensor_t', 'BTensor_t',
    'ThDataset_t', 'ThDataLoader_t',
    'ThNetwork_t', 'ThOptimizer_t',

    'Counter_t',
)
