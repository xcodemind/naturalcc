# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

from typing import *

import os
from collections import OrderedDict
from copy import deepcopy
from src.utils.util_gpu import occupy_gpu_new

# PORTIONs = [1.0, 0.8, 0.6, 0.4, 0.2, 0.01]
# PORTIONs = [  0.8, 0.6, 0.4, 0.2, 0.01]
# PORTIONs = [  0.001,0.0001]
# PORTIONs = [1.0, 0.8, 0.6, 0.4, 0.2, 0.01, 0.001, 0.0001]
# PORTIONs = [  0.8, 0.6, 0.4, 0.2, 0.01, 0.001, 0.0001]
# PORTIONs = [ 0 ]
# PORTIONs = [  0.001 ]



TRG_LNG = 'ruby'


def occupy_gpu(deivce=0, memory=10, ):
    print('PID -> {}, occpy GPU({}): {}G'.format(os.getpid(), deivce, memory))
    occupy_gpu_new(deivce, memory)

def code2seq(params ,PORTIONs ):
    new_params = deepcopy(params)
    new_params['lang_mode'] = 'unilang'
    new_params['method_name'] = 'code2seq'

    new_params['train_mode'] = 'train_sl'  # set 'None' as 1)train_prt, 2)train_ft
    # for src_lng in ['python', 'java', 'go', 'php', 'javascript', ]:
    # for src_lng in ['python', 'java',   'php', 'javascript', ]:

    for portion in PORTIONs:
        new_params['yaml'] = os.path.join('./yml_dir/ruby_code2seq_p{}_s.yml'.format(  portion))
        new_params['dataset_type'] = 'source'

        # before run, check each value is not None
        for name, value in new_params.items():
            assert value is not None, NotImplementedError(' {}\s value is None'.format(name))

        cmd = 'python -u ./run/main.py {}'.format(
            ' '.join(['--{} {}'.format(name, value) for name, value in new_params.items()])
        )
        print(cmd)
        os.system(cmd)


def ft(params,src_lng ,appendix=None ,PORTIONs=None ):
    new_params = deepcopy(params)
    new_params['lang_mode'] = 'xlang'
    new_params['method_name'] = 'finetune'

    new_params['train_mode'] = 'None'  # set 'None' as 1)train_prt, 2)train_ft
    # for src_lng in ['python', 'java', 'go', 'php', 'javascript', ]:
    # for src_lng in ['python', 'java',   'php', 'javascript', ]:

    for portion in PORTIONs:
        if appendix is None:
            new_params['yaml'] = os.path.join('./{}8{}/tok8path-p{}.yml'.format(src_lng, TRG_LNG, portion))
        else:
            new_params['yaml'] = os.path.join('./{}8{}_{}/tok8path-p{}.yml'.format(src_lng, TRG_LNG, appendix, portion))

        new_params['dataset_type'] = 'all'

        # before run, check each value is not None
        for name, value in new_params.items():
            assert value is not None, NotImplementedError('ft: {}\s value is None'.format(name))

        cmd = 'python -u ./run/main.py {}'.format(
            ' '.join(['--{} {}'.format(name, value) for name, value in new_params.items()])
        )
        print(cmd)
        os.system(cmd)


def dtrl(params,src_lng ,appendix=None ,PORTIONs=None ):
    new_params = deepcopy(params)
    new_params['lang_mode'] = 'xlang'
    new_params['method_name'] = 'dtrl'

    # train_dtrl_sl
    new_params['train_mode'] = 'None'
    # for src_lng in ['python', 'java', ]:
    for portion in PORTIONs:
        if appendix is not None:
            new_params['yaml'] = os.path.join('./{}8{}_{}/tok8path-p{}.yml'.format(src_lng, TRG_LNG,appendix, portion))
        else:
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


def main(src_lng,occupy,appendix=None,PORTIONs=None ):
    params = OrderedDict({
        'yaml': None,
        'task': 'summarization',
        'lang_mode': None,
        'method_name': None,
        'train_mode': None,
        'dataset_type': None,
        'log_root_dir':'/data/sjd/d/p_d/fse20all/100_small/',
        'occupy_gpu':'no'
    })



    ft(params=params,src_lng=src_lng,appendix= appendix,PORTIONs=PORTIONs)
    # dtrl(params,src_lng,appendix,PORTIONs)
    # code2seq(params,PORTIONs)

    device, gb = occupy.split('-')
    device = int(device)
    gb = float(gb)
    print('Occupying GPU. Enter Ctrl+C to complete. gpu: {} gb: {}'.format(device, gb))

    occupy_gpu_new(device, gb, compute=False)



if __name__ == '__main__':
    # python run/shell_s.py

    # # appendix = None
    appendix = 'pe1'

    # PORTIONs = [1.0, 0.8, 0.6, 0.4, 0.2, 0.01, 0.001, 0.0001]
    # PORTIONs = [ 0.8,0.2 ,0.001, 0.0001] # g5
    PORTIONs = [0.6, 0.4,  0.01  ] # g7
    # PORTIONs = [   0.2 ]
    # PORTIONs = [  0.01 ]

    # src_lng = 'python'
    # occupy ='3-5'

    # src_lng = 'php'
    # occupy = '5-10' # ft 10  dtrl 5.6

    # src_lng = 'javascript'
    # occupy = '1-7'

    # src_lng = 'java'
    # occupy = '7-5.2'

    # src_lng = 'javascript'
    # occupy = '5-7'

    # src_lng = 'go'
    # occupy = '5-10'

    src_lng = 'go'
    occupy = '7-10'

    print("src_lng:{}  occupy_gpu:{} TRG_LNG:{} PORTIONs :{} ".format(src_lng,occupy,TRG_LNG , PORTIONs))

    main(src_lng=src_lng, occupy=occupy, appendix=appendix, PORTIONs=PORTIONs)

################
    # python run/shell_s.py

    # appendix = 'lr4e-4'
    #
    # src_lng = 'python'
    # occupy ='0-5'

    # src_lng = 'php'
    # occupy = '2-5.6' # ft 10  dtrl 5.6

    # src_lng = 'javascript'
    # occupy = '3-7'

    # src_lng = 'java'
    # occupy = '6-5.2'


    # print("src_lng:{}  occupy_gpu:{} TRG_LNG:{} PORTIONs :{} ".format(src_lng,occupy,TRG_LNG , PORTIONs))
    # main(src_lng,occupy,appendix )

####### code2seq portion

    # occupy = '0-6'
    # print("code2seq occupy_gpu:{} TRG_LNG:{} PORTIONs :{} ".format( occupy,TRG_LNG , PORTIONs))
    # main(None,occupy  )

####
    # src_lng = 'java8javascript8php8python'
    # occupy = '0-10'
    # PORTIONs = [1.0, 0.0001]

    # src_lng = 'java8javascript8php8python'
    # occupy = '2-10'
    # PORTIONs = [0.8, 0.001]

    # src_lng = 'java8javascript8php8python'
    # occupy = '5-10'
    # PORTIONs = [0.6, 0.01]

    # src_lng = 'java8javascript8php8python'
    # occupy = '7-10'
    # PORTIONs = [0.2, 0.4]


    # print("src_lng:{}  occupy_gpu:{} TRG_LNG:{} PORTIONs :{} ".format(src_lng,occupy,TRG_LNG , PORTIONs))
    # main(src_lng=src_lng,occupy=occupy ,PORTIONs=PORTIONs )
####
    # appendix = 'pe1'
    #
    # # src_lng = 'go8java8javascript8php8python'
    # # occupy = '7-10'
    # # PORTIONs = [1.0, 0.0001 , 0.001 ]
    #
    # # src_lng = 'go8java8javascript8php8python'
    # # occupy = '0-10' # wanyao
    # # PORTIONs = [0.8,0.2]
    #
    #
    # src_lng = 'go8java8javascript8php8python'
    # occupy = '2-10'
    # PORTIONs = [0.6, 0.4, 0.01]
    #
    #
    # print("src_lng:{}  occupy_gpu:{} TRG_LNG:{} PORTIONs :{} ".format(src_lng,occupy,TRG_LNG , PORTIONs))
    # main(src_lng=src_lng,occupy=occupy, appendix=appendix  ,PORTIONs=PORTIONs )