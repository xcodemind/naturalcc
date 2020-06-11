# -*- coding: utf-8 -*-


import os
import wget
from ncc.types import *

MODES = ['train', 'valid', 'test']

path_join_func: Func_t = os.path.join
expanduser_func = os.path.expanduser
makedirs_func: Func_t = lambda dir: os.makedirs(dir, exist_ok=True)

download_func = lambda url, out: wget.download(url, out)  # download file at out directory/out file

__all__ = (
    'MODES',
    'path_join_func', 'expanduser_func', 'makedirs_func',

    'download_func',
)
