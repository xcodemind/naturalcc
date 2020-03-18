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

__all__ = [
    'os', 'sys',
    'Any', 'Dict', 'Tuple', 'List', 'Union', 'Optional',
    'LOGGER',
    'torch', 'nn', 'Module', 'F', 'DataLoader', 'Optimizer',
    'abc',
    'pprint', 'deepcopy', 'random', 'np', 'namedtuple',
    'TimeAnalyzer',
]
