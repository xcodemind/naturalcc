# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

from ncc.dataset.unilang_dataloader import UnilangDataloader
from ncc.dataset.xlang_dataloader import XlangDataloader
from ncc.dataset.kd_dataloader import KDDataloader

__all__ = [
    'UnilangDataloader', 'XlangDataloader',
    'KDDataloader',

]


# __all__ = [
#     'UnilangDataloader', 'XlangDataloader',
#
#
# ]