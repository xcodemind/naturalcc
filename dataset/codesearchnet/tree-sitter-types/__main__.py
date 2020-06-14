# -*- coding: utf-8 -*-

from typing import Dict
from ncc.types import *
from ncc.utils import path
from dataset.codesearchnet.utils.constants import LANGUAGES

TREE_SITTER_TYPES: Dict_t[String_t, Const_t] = {
    lng: 'https://raw.githubusercontent.com/tree-sitter/tree-sitter-{}/master/src/node-types.json'.format(lng)
    for lng in LANGUAGES
}

