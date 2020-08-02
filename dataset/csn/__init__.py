# -*- coding: utf-8 -*-


DATASET_NAME = 'CodeSearchNet'
RAW_DATA_DIR = '~/.ncc/{}/raw'.format(DATASET_NAME)
LIBS_DIR = '~/.ncc/{}/libs'.format(DATASET_NAME)
FLATTEN_DIR = '~/.ncc/{}/flatten'.format(DATASET_NAME)

LANGUAGES = ['ruby', 'python', 'java', 'go', 'php', 'javascript']
MODES = ['train', 'valid', 'test']

MEANINGLESS_TOKENS = set(['(', ')', '[', ']', '{', '}', ';', '@', '#', ':', '()', '<>', '{}'])
COMMENT_END_TOKENS = set(['{', '[', '('])

SBT_PARENTHESES = ['(_SBT', ')_SBT']

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)-15s] %(levelname)7s >> %(message)s (%(filename)s:%(lineno)d, %(funcName)s())',
    datefmt='%Y-%m-%d %H:%M:%S',
)
LOGGER = logging.getLogger(__name__)
