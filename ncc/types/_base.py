# -*- coding: utf-8 -*-

from typing import (
    Optional, Union, Generic,
    Sequence, List, Tuple,
    TypeVar,
    NoReturn,
    Any,
    Mapping, Dict,
    Set,
    Callable,
    Iterator, Iterable
)
from abc import ABC
from numba import (
    int8, uint8, int16, uint16, int32, uint32, int64, uint64,
    intc, uintc,  # c language int type
    float32, float64,
)

Any_t = Any
Const_t = Any
Void_t = Union[None, NoReturn]
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

# generic type
T = TypeVar('T')
S = TypeVar('S')
Sequence_t = Union[Sequence[T], List[T], Tuple[T]]  # for list & tuple
Dict_t = Union[Mapping[T, S], Dict[T, S]]
Set_t = Set
Iterator_t = Union[Iterator[T], Iterable[T]]

__all__ = (
    'String_t', 'Int_t', 'Float_t', 'Number_t', 'Bool_t', 'Byte_t',
    'Any_t', 'Const_t', 'Void_t', 'Func_t', 'Class_t', 'Exception_t',
    'Sequence_t', 'Dict_t', 'Set_t', 'Iterator_t',
    'Optional', 'Union', 'Tuple',
)
