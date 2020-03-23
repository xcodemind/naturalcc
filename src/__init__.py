# -*- coding: utf-8 -*-


import os
import sys

sys.path.append(os.path.abspath('.'))

from typing import Any, Dict, Tuple, List, Union, Optional

from pprint import pprint

import logging
from src.log import *

LOGGER = get_logger(level=logging.INFO)  # INFO, DEBUG
# LOGGER = get_logger(level=logging.DEBUG )  # INFO, DEBUG

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer

import abc

from copy import deepcopy

import random
import numpy as np
from collections import namedtuple

from line_profiler import LineProfiler

TimeAnalyzer = LineProfiler()

import src.criterions  # noqa

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

__all__ = [
    # 'pdb',
    'os', 'sys',
    'Any', 'Dict', 'Tuple', 'List', 'Union', 'Optional',
    'LOGGER',
    'torch', 'nn', 'Module', 'F', 'DataLoader', 'Optimizer',
    'abc',
    'pprint', 'deepcopy', 'random', 'np', 'namedtuple',
    'TimeAnalyzer',
    ]
__version__ = '0.9.0'

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
from src.log import meters, metrics, progress_bar  # noqa
sys.modules['src.meters'] = meters
sys.modules['src.metrics'] = metrics
sys.modules['src.progress_bar'] = progress_bar

import src.criterions  # noqa
import src.model  # noqa
import src.module  # noqa
import src.optim  # noqa
import src.optim.lr_scheduler  # noqa
import src.utils.pdb as pdb # noqa
import src.tasks  # noqa

# import src.benchmark  # noqa