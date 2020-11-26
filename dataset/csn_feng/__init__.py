# -*- coding: utf-8 -*-

import os
from dataset import (
    HOSTNAME, USERNAME, DEFAULT_DIR, LIBS_DIR, LOGGER,
)

DATASET_NAME = 'code_search_net_feng'
DATASET_DIR = os.path.join(DEFAULT_DIR, DATASET_NAME)
RAW_DATA_DIR = os.path.join(DATASET_DIR, 'raw')
FLATTEN_DIR = os.path.join(DATASET_DIR, 'flatten')
REFINE_DIR = os.path.join(DATASET_DIR, 'refine')
FILTER_DIR = os.path.join(DATASET_DIR, 'filter')

_RETRIEVAL_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, 'retrieval')

RETRIEVAL_RAW_DATA_DIR = {
    'train': os.path.join(_RETRIEVAL_RAW_DATA_DIR, 'train_valid'),
    'valid': os.path.join(_RETRIEVAL_RAW_DATA_DIR, 'train_valid'),
    'test': _RETRIEVAL_RAW_DATA_DIR,
}
RETRIEVAL_SPLITTER = '<CODESPLIT>'

LANGUAGES = ['ruby', 'python', 'java', 'go', 'php', 'javascript']
MODES = ['train', 'valid', 'test']

__all__ = (
    DATASET_NAME,
    RAW_DATA_DIR, LIBS_DIR, FLATTEN_DIR, REFINE_DIR, FILTER_DIR,
    LANGUAGES, MODES,
)
