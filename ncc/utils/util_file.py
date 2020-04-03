# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.path.abspath('.'))

'''
util_file.py
read/load gz, json, jsonline, yaml
'''

import itertools

from typing import Dict
import ruamel.yaml as yaml
from copy import deepcopy


def load_yaml(yaml_file: str) -> Dict:
    '''
    read yaml file
    :param yaml_file:
    '''
    with open(yaml_file, 'r', encoding='utf-8') as reader:
        return yaml.safe_load(reader)

# def load_args8yml(args):
#     yaml_file = os.path.join(sys.path[0], args.yaml)
#     print("load_args8yml yaml_file: ",yaml_file)
#     config = load_yaml(yaml_file)
#     config['dataset']['save_dir'] = args.save_dir
#     config['dataset']['dataset_dir'] = args.dataset_dir
#     config['common']['device'] = int(args.device)
#     if config.__contains__('kd'):
#         kd_path = args.save_dir.replace('result', 'kd_path')
#         if not os.path.exists(kd_path):
#             os.makedirs(kd_path)
#         config['kd']['kd_path'] = kd_path
#     return config

def load_args(yaml_file: str, args=None) -> Dict:
    # load transfer learning config for later load dataset
    if args is None:
        args = load_yaml(yaml_file)

    # add dict based on code_modalities
    args['dicts'] = {}
    lngs = []
    if args['dataset']['source'] is not None:
        lngs.extend(args['dataset']['source']['dataset_lng'])
    if args['dataset']['target'] is not None:
        lngs.extend(args['dataset']['target']['dataset_lng'])
    lngs = list(itertools.chain(*[k.split("8") for k in lngs]))
    lngs = sorted(list(set(lngs)))
    # pop path and substitute it with border/center
    dict_modalities = deepcopy(args['training']['code_modalities'])
    if 'path' in dict_modalities:
        del dict_modalities[dict_modalities.index('path')]
        dict_modalities.extend(['center', 'border'])
    dict_modalities.append('comment')
    for modal in dict_modalities:
        if len(lngs) > 1:
            args['dicts'][modal] = os.path.join(args['dataset']['dataset_dir'],
                                                  '{}.{}.dict'.format('_'.join(lngs), modal))
        elif len(lngs) == 1:
            args['dicts'][modal] = os.path.join(args['dataset']['dataset_dir'], lngs[0],
                                                  '{}.{}.dict'.format('_'.join(lngs), modal))
        else:
            raise NotImplementedError('dataset languages is None')

    # portion
    if args['dataset']['portion'] is None:
        args['dataset']['portion'] = 1.0

    return args


def load_args_kd(yaml_file: str, args=None ) -> Dict:
    # load transfer learning args for later load dataset
    if args is None:
        args = load_yaml(yaml_file)

    # add dict based on code_modalities
    args['dicts'] = {}

    # lngs = []
    # if args['dataset']['source'] is not None:
    #     lngs.extend(args['dataset']['source']['dataset_lng'])
    # if args['dataset']['target'] is not None:
    #     lngs.extend(args['dataset']['target']['dataset_lng'])
    # lngs = sorted(lngs)

    lngs = []
    if args['dataset']['source_domain']['source'] is not None:
        lngs.extend(args['dataset']['source_domain']['source']['dataset_lng'])
        lng_ss = args['dataset']['source_domain']['source']['dataset_lng']
    else:
        lng_ss = 'N'
    if args['dataset']['source_domain']['target'] is not None:
        lngs.extend(args['dataset']['source_domain']['target']['dataset_lng'])
        lng_st = args['dataset']['source_domain']['target']['dataset_lng']
    else:
        lng_st = 'N'
    if args['dataset']['target_domain']['source'] is not None:
        lngs.extend(args['dataset']['target_domain']['source']['dataset_lng'])
        lng_ts = args['dataset']['target_domain']['source']['dataset_lng']
    else:
        lng_ts = 'N'
    if args['dataset']['target_domain']['target'] is not None:
        lngs.extend(args['dataset']['target_domain']['target']['dataset_lng'])
        lng_tt = args['dataset']['target_domain']['target']['dataset_lng']
    else:
        lng_tt = 'N'
    lngs = sorted(lngs)

    args['kd']['all_lng'] = 'ss_'+'_'.join(lng_ss)+'_st_'+'_'.join(lng_st)+'_ts_'+'_'.join(lng_ts)+'_tt_'+'_'.join(lng_tt)
    # pop path and substitute it with border/center
    dict_modalities = deepcopy(args['training']['code_modalities'])
    if 'path' in dict_modalities:
        del dict_modalities[dict_modalities.index('path')]
        dict_modalities.extend(['center', 'border'])
    dict_modalities.append('comment')
    for modal in dict_modalities:
        if len(lngs) > 1:
            args['dicts'][modal] = os.path.join(args['dataset']['dataset_dir'],
                                                  '{}.{}.dict'.format('_'.join(lngs), modal))
        elif len(lngs) == 1:
            args['dicts'][modal] = os.path.join(args['dataset']['dataset_dir'], lngs[0],
                                                  '{}.{}.dict'.format('_'.join(lngs), modal))
        else:
            raise NotImplementedError('dataset languages is None')

    # portion
    if args['dataset']['portion'] is None:
        args['dataset']['portion'] = 1.0

    return args


if __name__ == '__main__':
    yaml_file = '/data/wanyao/yang/ghproj_d/GitHub/naturalcodev2/run/summarization/unilang/mm2seq/python-bak.yml'
    args = load_args(yaml_file)
    print(args)
