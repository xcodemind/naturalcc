# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.path.abspath('.'))

'''
util_file.py
read/load gz, json, jsonline, yaml
'''
from typing import Dict
import ruamel.yaml as yaml


def load_yaml(yaml_file: str) -> Dict:
    '''
    read yaml file
    :param yaml_file:
    '''

    def expanduser_for_dict(dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                expanduser_for_dict(value)

            if isinstance(value, str) and value.startswith('~/'):
                dictionary[key] = os.path.expanduser(value)
            elif isinstance(value, list):
                for i, val in enumerate(value):
                    if isinstance(val, str) and val.startswith('~/'):
                        value[i] = os.path.expanduser(val)

    with open(yaml_file, 'r', encoding='utf-8') as reader:
        args = yaml.safe_load(reader)
    expanduser_for_dict(args)
    return args
