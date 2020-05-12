# -*- coding: utf-8 -*-

LANGUAGES = ['ruby', 'python', 'java', 'go', 'php', 'javascript', ]
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

NO_METHOD = '<NO_METHOD>'

RECURSION_DEPTH = 999

MAX_AST_PATH_NUM = 300

MAX_AST_NODE_NUM = 5000
