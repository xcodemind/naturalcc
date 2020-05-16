# -*- coding: utf-8 -*-

MODES = ['train', 'valid', 'test']

PAD = "<pad>"
EOS = "</s>"
UNK = "<unk>"
BOS = "<s>"

# for bert
S_SEP = '<S_SEP>'
CLS = '<CLS>'
SNT_SEPS = [S_SEP, CLS]

# for path
H_SEP = '<H_SEP>'
T_SEP = '<T_SEP>'
P_SEP = '<P_SEP>'
PATH_SEPS = [H_SEP, T_SEP, S_SEP]

__all__ = [
    'MODES',
    'PAD', 'EOS', 'UNK', 'BOS',
    'H_SEP', 'T_SEP', 'PATH_SEPS',
    'S_SEP', 'CLS', 'SNT_SEPS',
]
