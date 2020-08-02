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
    with open(yaml_file, 'r', encoding='utf-8') as reader:
        args = yaml.safe_load(reader)
    # TODO: To be more elegant
    if 'preprocess' in args:
        # os.path.expanduser： ~/.ncc -> /home/user/.ncc
        for key, value in args['preprocess'].items():
            if isinstance(value, str) and value.startswith('~/'):
                args['preprocess'][key] = os.path.expanduser(value)
            if isinstance(value, list):
                for i, val in enumerate(value):
                    if val.startswith('~/'):
                        value[i] = os.path.expanduser(val)

    if 'task' in args:
        # os.path.expanduser： ~/.ncc -> /home/user/.ncc
        for key, value in args['task'].items():
            if isinstance(value, str) and value.startswith('~/'):
                args['task'][key] = os.path.expanduser(value)
            if isinstance(value, list):
                for i, val in enumerate(value):
                    if val.startswith('~/'):
                        value[i] = os.path.expanduser(val)

    if 'checkpoint' in args:
        # os.path.expanduser： ~/.ncc -> /home/user/.ncc
        for key, value in args['checkpoint'].items():
            if isinstance(value, str) and value.startswith('~/'):
                args['checkpoint'][key] = os.path.expanduser(value)
            if isinstance(value, list):
                for i, val in enumerate(value):
                    if val.startswith('~/'):
                        value[i] = os.path.expanduser(val)

    if 'eval' in args:
        # os.path.expanduser： ~/.ncc -> /home/user/.ncc
        for key, value in args['eval'].items():
            if isinstance(value, str) and value.startswith('~/'):
                args['eval'][key] = os.path.expanduser(value)
            if isinstance(value, list):
                for i, val in enumerate(value):
                    if val.startswith('~/'):
                        value[i] = os.path.expanduser(val)

    return args
