# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

from typing import *

import os
from collections import OrderedDict
from copy import deepcopy
from src.utils.util_gpu import occupy_gpu_new
from src.utils.constants import LANUAGES


def occupy_gpu(deivce=0, memory=10, ):
    print('PID -> {}, occpy GPU({}): {}G'.format(os.getpid(), deivce, memory))
    occupy_gpu_new(deivce, memory)


def nbow(params, ):
    for lng in LANUAGES:
        new_params = deepcopy(params)
        new_params['yaml'] = '{}.yml'.format(lng)

        cmd = 'nohup python -u ./run/retrieval/unilang/{}/main.py {} > ./run/retrieval/unilang/{}/retrieval_small_{}.log 2>&1 &'. \
            format(
            new_params['method_name'],
            ' '.join(['--{} {}'.format(name, value) for name, value in new_params.items()]),
            new_params['method_name'],
            lng
        )
        print(cmd)
        os.system(cmd)
        exit()


def main():
    params = OrderedDict({
        'yaml': None,
        'task': 'retrieval',
        'lang_mode': 'unilang',
        'method_name': 'nbow',
        'train_mode': 'all',
        'dataset_type': 'source',
    })

    nbow(params)
    # dtrl(params)
    occupy_gpu(memory=2)


if __name__ == '__main__':
    main()
