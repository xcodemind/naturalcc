# -*- coding: utf-8 -*-

MODES = ['train', 'valid', 'test']

PAD = "<pad>"
EOS = "</s>"
UNK = "<unk>"
BOS = "<s>"

# for code bert
S_SEP = '<S_SEP>'  # statement seperator
CLS = '<CLS>'
STATEMENT_SEPS = [S_SEP, CLS]
T_MASK = '<T_MASK>'  # token mask

# for path bert
H_SEP = '<H_SEP>'
T_SEP = '<T_SEP>'
P_SEP = '<P_SEP>'  # path seperator
PATH_SEPS = [H_SEP, T_SEP, P_SEP]
LN_MASK = '<LN_MASK>'  # leaf node mask
IN_MASK = '<IN_MASK>'  # intermediate node mask

# for unilm
SEP = '<SEP>'
S2S_SEP = '<S2S_SEP>'
S2S_BOS = '<S2S_BOS>'

# sentencepiece space tag for bep encoding
SP_SPACE = '▁'

# only for code modality in bert
INSERTED = '_inserted'

# tobe updated
# __all__ = [
#     'MODES',
#     'PAD', 'EOS', 'UNK', 'BOS',
#     'H_SEP', 'T_SEP', 'PATH_SEPS',
#     'S_SEP', 'CLS', 'STATEMENT_SEPS',
# ]
