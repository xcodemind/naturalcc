# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.path.abspath('.'))

'''
util_file.py
read/load gz, json, jsonline, yaml
'''

import itertools

from typing import *
import ruamel.yaml as yaml
from copy import deepcopy


def load_yaml(yaml_file: str) -> Dict:
    '''
    read yaml file
    :param yaml_file:
    '''
    with open(yaml_file, 'r', encoding='utf-8') as reader:
        return yaml.safe_load(reader)

def load_args8yml(args):
    yaml_file = os.path.join(sys.path[0], args.yaml)
    print("load_args8yml yaml_file: ",yaml_file)
    config = load_yaml(yaml_file)
    config['dataset']['save_dir'] = args.save_dir
    config['dataset']['dataset_dir'] = args.dataset_dir
    config['common']['device'] = int(args.device)
    if config.__contains__('kd'):
        kd_path = args.save_dir.replace('result', 'kd_path')
        if not os.path.exists(kd_path):
            os.makedirs(kd_path)
        config['kd']['kd_path'] = kd_path
    return config

def load_config(yaml_file: str,config=None) -> Dict:
    # load transfer learning config for later load dataset
    if config is None:
        config = load_yaml(yaml_file)

    # add dict based on code_modalities
    config['dicts'] = {}
    lngs = []
    if config['dataset']['source'] is not None:
        lngs.extend(config['dataset']['source']['dataset_lng'])
    if config['dataset']['target'] is not None:
        lngs.extend(config['dataset']['target']['dataset_lng'])
    lngs = list(itertools.chain(*[k.split("8") for k in lngs]))
    lngs = sorted(list(set(lngs)))
    # pop path and substitute it with border/center
    dict_modalities = deepcopy(config['training']['code_modalities'])
    if 'path' in dict_modalities:
        del dict_modalities[dict_modalities.index('path')]
        dict_modalities.extend(['center', 'border'])
    dict_modalities.append('comment')
    for modal in dict_modalities:
        if len(lngs) > 1:
            config['dicts'][modal] = os.path.join(config['dataset']['dataset_dir'],
                                                  '{}.{}.dict'.format('_'.join(lngs), modal))
        elif len(lngs) == 1:
            config['dicts'][modal] = os.path.join(config['dataset']['dataset_dir'], lngs[0],
                                                  '{}.{}.dict'.format('_'.join(lngs), modal))
        else:
            raise NotImplementedError('dataset languages is None')

    # portion
    if config['dataset']['portion'] is None:
        config['dataset']['portion'] = 1.0

    return config


def load_config_kd(yaml_file: str,config=None ) -> Dict:
    # load transfer learning config for later load dataset
    if config is None:
        config = load_yaml(yaml_file)

    # add dict based on code_modalities
    config['dicts'] = {}

    # lngs = []
    # if config['dataset']['source'] is not None:
    #     lngs.extend(config['dataset']['source']['dataset_lng'])
    # if config['dataset']['target'] is not None:
    #     lngs.extend(config['dataset']['target']['dataset_lng'])
    # lngs = sorted(lngs)

    lngs = []
    if config['dataset']['source_domain']['source'] is not None:
        lngs.extend(config['dataset']['source_domain']['source']['dataset_lng'])
        lng_ss = config['dataset']['source_domain']['source']['dataset_lng']
    else:
        lng_ss = 'N'
    if config['dataset']['source_domain']['target'] is not None:
        lngs.extend(config['dataset']['source_domain']['target']['dataset_lng'])
        lng_st = config['dataset']['source_domain']['target']['dataset_lng']
    else:
        lng_st = 'N'
    if config['dataset']['target_domain']['source'] is not None:
        lngs.extend(config['dataset']['target_domain']['source']['dataset_lng'])
        lng_ts = config['dataset']['target_domain']['source']['dataset_lng']
    else:
        lng_ts = 'N'
    if config['dataset']['target_domain']['target'] is not None:
        lngs.extend(config['dataset']['target_domain']['target']['dataset_lng'])
        lng_tt = config['dataset']['target_domain']['target']['dataset_lng']
    else:
        lng_tt = 'N'
    lngs = sorted(lngs)

    config['kd']['all_lng'] = 'ss_'+'_'.join(lng_ss)+'_st_'+'_'.join(lng_st)+'_ts_'+'_'.join(lng_ts)+'_tt_'+'_'.join(lng_tt)
    # pop path and substitute it with border/center
    dict_modalities = deepcopy(config['training']['code_modalities'])
    if 'path' in dict_modalities:
        del dict_modalities[dict_modalities.index('path')]
        dict_modalities.extend(['center', 'border'])
    dict_modalities.append('comment')
    for modal in dict_modalities:
        if len(lngs) > 1:
            config['dicts'][modal] = os.path.join(config['dataset']['dataset_dir'],
                                                  '{}.{}.dict'.format('_'.join(lngs), modal))
        elif len(lngs) == 1:
            config['dicts'][modal] = os.path.join(config['dataset']['dataset_dir'], lngs[0],
                                                  '{}.{}.dict'.format('_'.join(lngs), modal))
        else:
            raise NotImplementedError('dataset languages is None')

    # portion
    if config['dataset']['portion'] is None:
        config['dataset']['portion'] = 1.0

    return config


if __name__ == '__main__':
    yaml_file = '/data/wanyao/yang/ghproj_d/GitHub/naturalcodev2/run/summarization/unilang/mm2seq/python-bak.yml'
    config = load_config(yaml_file)
    print(config)
