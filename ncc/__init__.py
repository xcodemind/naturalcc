# -*- coding: utf-8 -*-


# import os
# import sys
#
# # sys.path.append(os.path.abspath('.'))
#
# from typing import Any, Dict, Tuple, List, Union, Optional
#
# from pprint import pprint

import logging

# from ncc.logging import get_logger


# LOGGER = get_logger()  # INFO, DEBUG
# LOGGER = get_logger(level=logging.DEBUG )  # INFO, DEBUG
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)-15s] %(levelname)7s >> %(message)s (%(filename)s:%(lineno)d, %(funcName)s())',
    datefmt='%Y-%m-%d %H:%M:%S',
)
LOGGER = logging.getLogger(__name__)
# import torch
# from torch import nn
# from torch.nn import Module
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torch.optim.optimizer import Optimizer
#
# import abc
#
# from copy import deepcopy
#
# import random
# import numpy as np
# from collections import namedtuple

# from line_profiler import LineProfiler

# TimeAnalyzer = LineProfiler()

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
__all__ = [
    # 'pdb',
    # 'os', 'sys',
    # 'Any', 'Dict', 'Tuple', 'List', 'Union', 'Optional',
    'LOGGER',
    # 'torch', 'nn', 'Module', 'F', 'DataLoader', 'Optimizer',
    # 'abc',
    # 'pprint', 'deepcopy', 'random', 'np', 'namedtuple',
    # 'TimeAnalyzer',
    # '__version__',
]

# __all__ = [
#     'os', 'sys',
#     'Any', 'Dict', 'Tuple', 'List', 'Union', 'Optional',
#     'LOGGER',
#     'torch', 'nn', 'Module', 'F', 'DataLoader', 'Optimizer',
#     'abc',
#     'pprint', 'deepcopy', 'random', 'np', 'namedtuple',
#     'TimeAnalyzer',
# ]
import sys

# backwards compatibility to support `from fairseq.meters import AverageMeter`
from ncc.logging import meters, metrics, progress_bar  # noqa

sys.modules['ncc.meters'] = meters
sys.modules['ncc.metrics'] = metrics
sys.modules['ncc.progress_bar'] = progress_bar

# import ncc.criterions  # noqa
# import ncc.model  # noqa
# import ncc.modules  # noqa
# import ncc.optim  # noqa
# import ncc.optim.lr_scheduler  # noqa
# import ncc.utils.pdb as pdb # noqa
# import ncc.tasks  # noqa

# import ncc.benchmark  # noqa

# ============== NCC Library info ============== #
# library properties
__NAME__ = 'NCC'
__FULL_NAME__ = 'Natural Code and Comment Lib.'
__DESCRIPTION__ = '''
NCC (Natural Code and Comment Lib.) is developed for Programming Language Processing and Code Analysis.
'''
# __VERSION__ = ‘x,y(a,b,c)’, e.g. '1.2c'
# x for framework update or a considerable update,
# y for internal update, e.g. naturalcodev3 is our 3rd Version of coding,
__VERSION__ = '0.3'
__AUTHORS__ = [
    'YangHE', 'YaoWAN',  # main authors, sharing same contributions
    # other authors
    'JingdongSUN', 'WeiHUA',
]
__AFFILIATIONS__ = [
    'Univ. of Tech. Sydney (UTS)', 'Z.J. Univ. (ZJU)',  # main affiliations, sharing same contributions
    # other affiliations
    'Wuhan Univ. of Tech. (WHUT)',
]
__BIRTH__ = '2018/06-Now'

import os

__ROOT_DIR__ = '~/.ncc'
__CACHE_DIR__ = '~/.ncc/cache'
# ============== NCC Library info ============== #
