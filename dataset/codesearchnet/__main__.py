# -*- coding: utf-8 -*-

from typing import *

import os
from multiprocessing import cpu_count
# from .codesearchnet import CodeSearchNet
from dataset.codesearchnet.codesearchnet import CodeSearchNet


def main():
    print('Current PID: {}'.format(os.getpid()))
    # download neccesary files
    dataset = CodeSearchNet(download=True, thread_num=cpu_count())

    lngs = ['ruby']
    flatten_attrs = ['code', 'docstring']
    # flatten_attrs = ['code_tokens', 'docstring_tokens']
    # tree_attrs = ['path']
    # all_attrs = flatten_attrs + tree_attrs
    # flatten raw files separately
    dataset.flatten_data_all(lngs, save_attrs=flatten_attrs, overwrite=True)
    # parse raw_ast into other new tree modalities
    # dataset.parse_new_tree_modalities_all(lngs, modalities=tree_attrs, overwrite=True)
    # use cat command to merge files
    dataset.merge_attr_files(lngs, attrs=flatten_attrs)


if __name__ == '__main__':
    main()
