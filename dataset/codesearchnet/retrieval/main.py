# -*- coding: utf-8 -*-

import os
from multiprocessing import cpu_count, Pool
from dataset.codesearchnet.retrieval import parse_data


def main():
    raw_dir = '/data/wanyao/yang/ghproj_d/GitHub/datasetv2/ase2020_retrieval'
    os.makedirs(raw_dir, exist_ok=True)
    ################################################################
    # use all cores
    ################################################################
    mpool = Pool(cpu_count())
    parse_data.flatten_raw_data(mpool, raw_dir)
    parse_data.parse_ast_modalities(mpool, raw_dir)

    from dataset.parse_retrieval.dicts import main as dict_main
    dict_main(dataset_dir=raw_dir, KEYS=None, xlang=False, )


if __name__ == '__main__':
    main()
