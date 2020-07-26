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

    lngs = ['java']
    flatten_attrs = ['code', 'raw_ast']
    # flatten raw files separately
    dataset.flatten_data_all(lngs, save_attrs=flatten_attrs, overwrite=True)
    dataset.merge_attr_files(lngs, attrs=flatten_attrs)  # use cat command to merge files

    # parse raw_ast into other new tree modalities
    tree_attrs = ['new_ast']
    dataset.parse_new_tree_modalities_all(lngs, modalities=tree_attrs, overwrite=True)
    dataset.merge_attr_files(lngs, attrs=tree_attrs)


if __name__ == '__main__':
    main()
