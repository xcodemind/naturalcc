# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

from src.data.dict import Dict as _Dict
from src.data.token_dicts import TokenDicts

__all__ = [
    '_Dict', 'TokenDicts',
]
