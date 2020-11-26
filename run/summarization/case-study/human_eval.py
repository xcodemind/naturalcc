# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

from typing import *
import numpy as np


def parse(info):
    info = [line.strip().split('\t') for line in info.split('\n')]
    new_info = {line[0]: np.asarray([int(score) for score in line[1:]]) for line in info}  # +1 for avoid 0
    return new_info


def eval(infos):
    models = list(infos[0].keys())
    human_score = {}

    for model in models:
        human_score[model] = (np.sum([info[model] for info in infos], axis=0) + 1).mean()

    return human_score

# GT: 1 1 1 0 1

def main():
    hy = \
        '''
        GT	1	1	1	0	1	1	1	1	1	1	1	1	1	1	1	1	1	0	0	1	1	1	0	1	0	1	1	1	1	1	0	1	1	1	0	1	1	1	1	0	1	1	0	1	0	0	1	0	0	1	1	1	1	0	1	0	1	1	1	1	0	1	1	1	1	0	1	0	1	0	1	1	1	0	1	1
        code2seq	0	0	0	0	0	1	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	1	0	0	0	1	0	0	0	0	1	0	0	1	0	1	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	1	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0
        ft	0	0	0	0	0	0	1	1	0	0	0	1	0	0	1	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	1	0	0	0	0	1	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	1	1
        maml	1	1	1	0	1	0	0	0	0	1	0	1	0	1	1	1	1	0	0	1	0	0	1	0	0	1	0	0	1	0	0	1	1	0	0	0	0	0	0	0	1	0	0	0	1	0	0	1	1	1	1	0	0	0	1	0	0	0	0	0	1	1	0	1	1	0	0	0	0	1	1	1	0	1	1	1
        mm2seq	0	0	0	0	1	0	1	0	1	0	0	0	0	1	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	1	1	0	0	1	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	1	0	1	1	0	0
        seq2seq	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
        '''.strip()

    jd = \
        '''
        GT	1	1	1	0	1	1	1	1	1	1	1	1	1	1	1	1	1	0	0	1	1	1	0	1	0	1	1	1	1	1	0	1	1	1	0	1	1	1	1	0	1	1	0	1	0	0	1	0	0	1	1	1	1	0	1	0	1	1	1	1	0	1	1	1	1	0	1	0	1	0	1	1	1	0	1	1
        code2seq	0	0	0	0	0	1	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	1	0	0	0	1	0	0	0	0	1	0	0	0	0	0	0	0	1	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	1	1	0	0	0	0	0	0	0	0	0	0	0	0	0
        ft	0	0	0	0	0	0	0	0	1	0	0	1	0	0	1	0	0	0	0	0	0	0	1	0	1	0	0	0	0	1	0	0	0	0	0	1	0	0	0	1	0	1	0	1	0	0	1	0	1	0	1	0	0	0	0	0	0	0	1	1	0	0	1	0	0	0	0	0	0	0	1	0	0	1	1	1
        maml	1	1	1	0	0	1	0	0	0	0	0	0	1	1	1	1	1	1	1	1	1	0	1	0	0	1	0	0	1	1	0	1	1	1	0	1	1	1	0	1	1	1	1	1	0	1	1	1	1	1	1	1	0	1	0	1	1	1	1	1	1	1	0	1	1	0	1	0	1	1	0	1	1	1	1	1
        mm2seq	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	1	0	1	1	0	0	0	0	0	0	0	0	0	1	0	0	0	1	0	0	0	0	0	0	0	1	0	0	0	0	1	1	1	0	0	1	1	1	0	0	1	0	1	1	0	0	0	0	0	1	1	0	1	1	1	1	1	0
        seq2seq	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
        '''.strip()

    hy_score = parse(hy)
    jd_score = parse(jd)
    scores = eval([hy_score, jd_score])
    print(scores)


if __name__ == '__main__':
    main()
