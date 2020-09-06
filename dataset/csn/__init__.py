# -*- coding: utf-8 -*-

import os

DATASET_NAME = 'code_search_net'
RAW_DATA_DIR = '~/.ncc/{}/raw'.format(DATASET_NAME)
LIBS_DIR = '~/.ncc/{}/libs'.format(DATASET_NAME)
FLATTEN_DIR = '~/.ncc/{}/flatten'.format(DATASET_NAME)
REFINE_DIR = '~/.ncc/{}/refine'.format(DATASET_NAME)
FILTER_DIR = '~/.ncc/{}/filter'.format(DATASET_NAME)

RAW_DATA_DIR, LIBS_DIR, FLATTEN_DIR, REFINE_DIR, FILTER_DIR = \
    map(os.path.expanduser, (RAW_DATA_DIR, LIBS_DIR, FLATTEN_DIR, REFINE_DIR, FILTER_DIR))

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

MEANINGLESS_TOKENS = set(['(', ')', '[', ']', '{', '}', ';', '@', '#', ':', '()', '<>', '{}'])
COMMENT_END_TOKENS = set(['{', '[', '('])

import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)-15s] %(levelname)7s >> %(message)s (%(filename)s:%(lineno)d, %(funcName)s())',
    datefmt='%Y-%m-%d %H:%M:%S',
)
LOGGER = logging.getLogger(__name__)
