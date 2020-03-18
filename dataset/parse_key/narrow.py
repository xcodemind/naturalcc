# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

from dataset.utils.util_ast import *
from src.utils.constants import *
import ujson
import glob

import random

random.seed(666)
import numpy as np

np.random.seed(666)


def data_len(lng: str) -> Dict:
    key = 'index'
    tok_size = {}
    for mode in MODES:
        tok_size[mode] = []
        for index_filename in sorted([
            filename for filename in glob.glob('{}/*'.format(os.path.join(dataset_dir, lng, key)))
            if mode in filename
        ]):
            tok_filename = index_filename.replace('index', 'tok')
            with open(index_filename, 'r') as index_reader:
                with open(tok_filename, 'r') as tok_reader:
                    index = index_reader.readline().strip()
                    tok = tok_reader.readline().strip()
                    while len(index) > 0 and len(tok) > 0:
                        tok_size[mode].append([ujson.loads(index), len(ujson.loads(tok))])
                        index = index_reader.readline().strip()
                        tok = tok_reader.readline().strip()

    return tok_size


def copy(src_dir: str, lng: str, src_file: str,
         dst_dir: str, DICT_KEYS: List, index_set, ):
    src_dir = os.path.join(src_dir, lng)
    src_filenames = {key: os.path.join(src_dir, key, src_file) for key in DICT_KEYS}
    src_readers = {key: open(src_file, 'r') for key, src_file in src_filenames.items()}
    for key in DICT_KEYS:
        os.makedirs(os.path.join(dst_dir, lng, key), exist_ok=True)
    dst_filenames = {key: os.path.join(dst_dir, lng, key, src_file) for key in DICT_KEYS}
    dst_writers = {key: open(src_file, 'w') for key, src_file in dst_filenames.items()}

    LOGGER.info('copy from {} to {}'.format(
        os.path.join(src_dir, '*', src_file),
        os.path.join(dst_dir, lng, '*', src_file)
    ))

    info = {key: src_readers[key].readline().strip() for key in DICT_KEYS}
    while len(info['index']) > 0:
        index = ujson.loads(info['index'])
        if index in index_set:
            for key in DICT_KEYS:
                dst_writers[key].write(info[key] + '\n')

        info = {key: src_readers[key].readline().strip() for key in DICT_KEYS}

    # close
    for key in DICT_KEYS:
        src_readers[key].close()
        dst_writers[key].close()


def narrow(paralleler, src_dir: str, dst_dir: str, KEYS: List, ):
    LNGS_NUM = {
        "python": {"train": 1000, "valid": 200, "test": 500},
        "java": {"train": 1000, "valid": 200, "test": 500},
        "go": {"train": 1000, "valid": 200, "test": 500},
        "php": {"train": 1000, "valid": 200, "test": 500},
        "ruby": {"train": 1000, "valid": 200, "test": 500},
        "javascript": {"train": 1000, "valid": 200, "test": 500},
    }

    params = []
    for lng in LANUAGES:
        tok_size = data_len(lng)

        for mode in MODES:
            index_tok_size = sorted(tok_size[mode], key=lambda pair: pair[-1])[:LNGS_NUM[lng][mode]]
            index, _ = zip(*index_tok_size)
            index_set = set(index)

            filenames = sorted([filename for filename in os.listdir(os.path.join(src_dir, lng, 'index'))
                                if mode in filename])
            for filename in filenames:
                params.append([src_dir, lng, filename,
                               dst_dir, KEYS, deepcopy(index_set)])
    # copy(*params[0])
    paralleler(delayed(copy)(*param) for param in params)


def main():
    ################################################################
    # use all cores
    ################################################################
    from multiprocessing import cpu_count
    paralleler = Parallel(n_jobs=cpu_count())  # build a multi-processing pool

    dataset_dir = '/data/wanyao/yang/ghproj_d/GitHub/datasetv2/key/100'
    # dst_dir = '/data/wanyao/yang/ghproj_d/GitHub/datasetv2/key/100_small'
    # DICT_KEYS = ['tok', 'code', 'code_tokens', 'sbt', 'sbtao', 'sbt2', 'comment', 'docstring', 'docstring_tokens',
    #              'method', 'ast', 'path', 'index', 'func_name', 'raw_ast', ]

    dst_dir = '/data/wanyao/yang/ghproj_d/GitHub/datasetv2/key/demo'
    DICT_KEYS = ['tok', 'code', 'code_tokens', 'comment', 'docstring', 'docstring_tokens', 'path', 'index', ]

    narrow(paralleler, dataset_dir, dst_dir, DICT_KEYS, )

    from dataset.parse_key.dicts import main as dict_main
    dict_main(
        dataset_dir=dst_dir,
        KEYS=None,
        xlang=True,
    )


if __name__ == '__main__':
    main()
