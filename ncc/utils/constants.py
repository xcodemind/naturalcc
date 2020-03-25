# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.path.abspath('.'))

LANUAGES = ['python', 'java', 'go', 'php', 'ruby', 'javascript']
MODES = ['train', 'valid', 'test']

POS_INF = 999999999999999999
NEG_INF = -POS_INF
EPS_ZERO = 1e-12

PAD = 0  # must be 0, can't be changed
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<PAD>'
UNK_WORD = '<UNK>'
BOS_WORD = '<BOS>'
EOS_WORD = '<EOS>'

NODE_FIX = 'NODEFIX'
DGLGraph_PAD_WORD = -1

CODE_MODALITIES = ['seq', 'sbt', 'tree', 'cfg']

METRICS = ['bleu', 'meteor', 'rouge', 'cider']
