# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

from typing import *

import os
from collections import OrderedDict
from copy import deepcopy
from ncc.utils.util_gpu import occupy_gpu_new

PORTIONs = [1.0, 0.8, 0.6, 0.4, 0.2, 0.01, 0.001, 0.0001]
TRG_LNG = 'ruby'


def occupy_gpu(deivce=0, memory=10, ):
    print('PID -> {}, occpy GPU({}): {}G'.format(os.getpid(), deivce, memory))
    occupy_gpu_new(deivce, memory)


def ft(params, ):
    new_params = deepcopy(params)
    new_params['lang_mode'] = 'xlang'
    new_params['method_name'] = 'finetune'

    new_params['train_mode'] = 'None'  # set 'None' as 1)train_prt, 2)train_ft
    # for src_lng in ['python', 'java', 'go', 'php', 'javascript', ]:
    for src_lng in ['python', ]:
        for portion in [1.0, ]:
            new_params['yaml'] = os.path.join('./{}8{}/tok8path-p{}.yml'.format(src_lng, TRG_LNG, portion))
            new_params['dataset_type'] = 'all'

            # before run, check each value is not None
            for name, value in new_params.items():
                assert value is not None, NotImplementedError('DTRL: {}\s value is None'.format(name))

            cmd = 'python -u ./run/main.py {}'.format(
                ' '.join(['--{} {}'.format(name, value) for name, value in new_params.items()])
            )
            print(cmd)
            os.system(cmd)


def dtrl(params, ):
    new_params = deepcopy(params)
    new_params['lang_mode'] = 'xlang'
    new_params['method_name'] = 'dtrl'

    # train_dtrl_sl
    commands = []
    new_params['train_mode'] = 'train_dtrl_sl'
    for src_lng in ['javascript', ]:
        for portion in PORTIONs:
            new_params['yaml'] = os.path.join('./{}8{}/tok8path-p{}.yml'.format(src_lng, TRG_LNG, portion))
            new_params['dataset_type'] = 'all'

            # before run, check each value is not None
            for name, value in new_params.items():
                assert value is not None, NotImplementedError('DTRL: {}\s value is None'.format(name))

            cmd = 'python -u ./run/main.py {}'.format(
                ' '.join(['--{} {}'.format(name, value) for name, value in new_params.items()])
            )
            print(cmd)
            os.system(cmd)
            commands.append(cmd)

    # from joblib import Parallel, delayed
    # Parallel(n_jobs=3)(delayed(os.system)(cmd) for cmd in commands)


def main():
    params = OrderedDict({
        'yaml': None,
        'task': 'summarization',
        'lang_mode': None,
        'method_name': None,
        'train_mode': None,
        'dataset_type': None,
    })

    # ft(params)
    # dtrl(params)
    occupy_gpu(memory=10)


if __name__ == '__main__':
    main()
