# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.path.abspath('.'))

DATASET_DIR = '/data/wanyao/yang/ghproj_d/GitHub/datasetv2'
RAW_DATASET_DIR = os.path.join(DATASET_DIR, 'raw')
AST_NODE_CEILING = 200
KEY_DST_DIR = os.path.join(DATASET_DIR, str(AST_NODE_CEILING))
PATH_K = 300

SO_FILE = '/data/wanyao/yang/ghproj_d/GitHub/naturalcodev2/dataset/parser_zips/languages.so'

RAW_KEYS = ['code', 'code_tokens', 'docstring', 'docstring_tokens', 'func_name', 'original_string',
            'path', 'repo', 'sha', 'url']
SAVE_KEYS = ['code', 'code_tokens', 'docstring', 'docstring_tokens', 'func_name', ]
POP_KEYS = ['original_string', 'path', 'repo', 'sha', 'url', 'language', 'partition']

LANGUAGES = ['python', 'java', 'go', 'php', 'ruby', 'javascript']
MODES = ['train', 'valid', 'test']

# POP_KEYS = ['repo', 'path', 'language', 'original_string', 'partition', 'sha', 'url']

MEANINGLESS_TOKENS = set(['(', ')', '[', ']', '{', '}', ';', '@', '#', ':', '()', '<>', '{}'])
COMMENT_END_TOKENS = set(['{', '[', '('])
SBT_PARENTHESES = ['(_SBT', ')_SBT']

MAX_COMMENT_TOKEN_LIST_LEN = 25
MAX_RAW_COMMENT_LEN = 4
MAX_CODE_TOKEN_LEN = 70  # if the length is bigger than this threshold, skip this code snippet

MAX_TOKEN_SIZE = 50000

NODE_FIX = 'NODEFIX'  # 1 -> inf
ROOT_NODE_NAME = NODE_FIX + str(1)
NODE_TMP = 'TMP'
PAD_WORD = '<PAD>'

PAD_TOKEN_IND = 0
UNK_TOKEN_IND = 1
VALID_VOCAB_START_IND = 2
