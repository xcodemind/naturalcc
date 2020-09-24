# -*- coding: utf-8 -*-

import os
from dataset import (
    HOSTNAME, USERNAME, DEFAULT_DIR, LOGGER,
)

DATASET_NAME = 'code_search_net_feng'
DATASET_DIR = os.path.join(DEFAULT_DIR, DATASET_NAME)
RAW_DATA_DIR = os.path.join(DATASET_DIR, 'raw')
LIBS_DIR = os.path.join(DEFAULT_DIR, 'libs')
FLATTEN_DIR = os.path.join(DATASET_DIR, 'flatten')
REFINE_DIR = os.path.join(DATASET_DIR, 'refine')
FILTER_DIR = os.path.join(DATASET_DIR, 'filter')

RAW_DATA_DIR, LIBS_DIR, FLATTEN_DIR, REFINE_DIR, FILTER_DIR = \
    map(os.path.expanduser, (RAW_DATA_DIR, LIBS_DIR, FLATTEN_DIR, REFINE_DIR, FILTER_DIR))

_RETRIEVAL_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, 'retrieval')

RETRIEVAL_RAW_DATA_DIR = {
    'train': os.path.join(_RETRIEVAL_RAW_DATA_DIR, 'train_valid'),
    'valid': os.path.join(_RETRIEVAL_RAW_DATA_DIR, 'train_valid'),
    'test': _RETRIEVAL_RAW_DATA_DIR,
}
RETRIEVAL_SPLITTER = '<CODESPLIT>'

LANGUAGES = ['ruby', 'python', 'java', 'go', 'php', 'javascript']
MODES = ['train', 'valid', 'test']

RECURSION_DEPTH = 999  # dfs recursion limitation
# path modality
PATH_NUM = 300  # path modality number
# sbt modality
MAX_SUB_TOKEN_LEN = 5  # we only consider the first 5 sub-tokens from tokenizer
SBT_PARENTHESES = ['(_SBT', ')_SBT']
# for binary-AST
NODE_TMP = 'TMP'

__all__ = (
    DATASET_NAME,
    RAW_DATA_DIR, LIBS_DIR, FLATTEN_DIR, REFINE_DIR, FILTER_DIR,
    LANGUAGES, MODES,
    RECURSION_DEPTH, PATH_NUM, MAX_SUB_TOKEN_LEN, SBT_PARENTHESES, NODE_TMP,
)
