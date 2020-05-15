# -*- coding: utf-8 -*-

PAD = "<pad>"
EOS = "</s>"
UNK = "<unk>"
BOS = "<s>"
# for path
H_SEP = '<H_SEP>'
T_SEP = '<T_SEP>'
P_SEP = '<P_SEP>'
PATH_SEPS = [H_SEP, T_SEP, P_SEP]
# for bert
S_SEP = '<S_SEP>'
CLS = '<CLS>'

__all__ = [
    'PAD', 'EOS', 'UNK', 'BOS',
    'H_SEP', 'T_SEP', 'P_SEP', 'PATH_SEPS',
    'S_SEP', 'CLS',
]
