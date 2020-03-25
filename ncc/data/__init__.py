# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

from ncc.data.dict import Dict as _Dict
from ncc.data.token_dicts import TokenDicts
from ncc.data.dictionary import Dictionary
__all__ = [
    '_Dict', 'TokenDicts', 'Dictionary'
]
