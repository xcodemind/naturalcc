# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

from typing import *

import os
import glob
import ujson
import itertools
from pprint import pprint
from collections import Counter
from ncc.utils.constants import LANUAGES, MODES
import gzip
import jsonlines

# dataset_dir = '/data/wanyao/yang/ghproj_d/GitHub/datasetv2/key/100'
dataset_dir = '/data/wanyao/ghproj_d/naturalcodev2/datasetv2/base'
# dataset_dir = '/data/wanyao/yang/ghproj_d/GitHub/datasetv2/key/100_small'
save_dir = '/data/wanyao/yang/ghproj_d/GitHub/naturalcodev2/exp/basic_info/'
save_dir = os.path.join(save_dir, dataset_dir.split('/')[-1])
os.makedirs(save_dir, exist_ok=True)


def load_file(filename: str) -> List:
    with open(filename, 'r') as reader:
        return [ujson.loads(line.strip()) for line in reader.readlines() if len(line.strip()) > 0]


def load_jsonl_gz(filename: str):
    with gzip.open(filename, mode='rt') as reader:
        lines = reader.readlines()
        return jsonlines.Reader(lines)


def data_files(key: str) -> Dict:
    dataset_files = {
        lng: {
            mode: sorted([
                filename for filename in glob.glob('{}/*'.format(os.path.join(dataset_dir, lng, key)))
                if mode in filename
            ])
            for mode in MODES
        }
        for lng in LANUAGES
    }
    return dataset_files


def data_info(key: str):
    code_files = data_files(key)
    node_size = {}
    avg_size = {}
    data_len = {}
    for lng, files in code_files.items():
        node_size[lng] = {}
        avg_size[lng] = {}
        data_len[lng] = {}

        for mode, fls in files.items():
            data = list(itertools.chain(*[load_file(fl) for fl in fls]))

            if key == 'path':
                size = list(itertools.chain(*[
                    [len(leaf_path[0]) + len(leaf_path[1]) + len(leaf_path[2]) for leaf_path in line]
                    for line in data]))
            else:
                size = [len(line) for line in data]

            del data
            avg_size[lng][mode] = sum(size) / len(size)
            node_size[lng][mode] = dict(Counter(size))
            data_len[lng][mode] = len(size)
    return data_len, node_size, avg_size,


def raw_data_info():
    dataset_files = {
        lng: {
            mode: sorted([
                filename for filename in glob.glob('{}/*.jsonl.gz'.format(os.path.join(dataset_dir, lng, )))
                if mode in filename
            ])
            for mode in MODES
        }
        for lng in LANUAGES
    }
    print(dataset_files)

    tok_size = {}
    tok_avg_size = {}
    tree_size = {}
    ast_avg_size = {}
    comment_size = {}
    comment_avg_size = {}
    for lng, files in dataset_files.items():
        tok_size[lng] = {}
        tok_avg_size[lng] = {}
        tree_size[lng] = {}
        ast_avg_size[lng] = {}
        comment_size[lng] = {}
        comment_avg_size[lng] = {}

        for mode, fls in files.items():
            print('load {}'.format(fls))
            data = list(itertools.chain(*[load_jsonl_gz(fl) for fl in fls]))

            tmp_tok_size, tmp_tree_szie, tmp_comment_size = [], [], []
            for line in data:
                tmp_tok_size.append(len(line['tok']))
                tmp_tree_szie.append(len(line['tree']))
                tmp_comment_size.append(len(line['comment']))

            tok_size[lng][mode] = dict(Counter(tmp_tok_size))
            tok_avg_size[lng][mode] = sum(tmp_tok_size) / len(tmp_tok_size)

            tree_size[lng][mode] = dict(Counter(tmp_tree_szie))
            ast_avg_size[lng][mode] = sum(tmp_tree_szie) / len(tmp_tree_szie)

            comment_size[lng][mode] = dict(Counter(tmp_comment_size))
            comment_avg_size[lng][mode] = sum(tmp_comment_size) / len(tmp_comment_size)

    with open(os.path.join(save_dir, 'tok_size.dict'), 'w') as writer:
        writer.write(ujson.dumps(tok_size))
    with open(os.path.join(save_dir, 'tok_avg_size.dict'), 'w') as writer:
        writer.write(ujson.dumps(tok_avg_size))

    with open(os.path.join(save_dir, 'tree_size.dict'), 'w') as writer:
        writer.write(ujson.dumps(tree_size))
    with open(os.path.join(save_dir, 'ast_avg_size.dict'), 'w') as writer:
        writer.write(ujson.dumps(ast_avg_size))

    with open(os.path.join(save_dir, 'comment_size.dict'), 'w') as writer:
        writer.write(ujson.dumps(comment_size))
    with open(os.path.join(save_dir, 'comment_avg_size.dict'), 'w') as writer:
        writer.write(ujson.dumps(comment_avg_size))


def expore_dataset():
    '''
        language    partition， 所有的raw数据
        go test: 14291
        go train: 317832
        go valid: 14242
        java test: 26909
        java train: 454451
        java valid: 15328
        javascript test: 6483
        javascript train: 123889
        javascript valid: 8253
        php test: 28391
        php train: 523712
        php valid: 26015
        python test: 22176
        python train: 412178
        python valid: 23107
        ruby test: 2279
        ruby train: 48791
        ruby valid: 2209
        '''
    raw_data_info()
    # data_len, tok_size, tok_avg_size, ast_size, ast_avg_size, path_size, path_avg_size = raw_data_info(keys)

    # data_len, tok_size, tok_avg_size, = data_info(key='tok')
    # with open(os.path.join(save_dir, 'data_len.dict'), 'w') as writer:
    #     writer.write(ujson.dumps(data_len))
    # with open(os.path.join(save_dir, 'tok_size.dict'), 'w') as writer:
    #     writer.write(ujson.dumps(tok_size))
    # with open(os.path.join(save_dir, 'tok_avg_size.dict'), 'w') as writer:
    #     writer.write(ujson.dumps(tok_avg_size))
    #
    # _, ast_size, ast_avg_size, = data_info(key='ast')
    # with open(os.path.join(save_dir, 'ast_size.dict'), 'w') as writer:
    #     writer.write(ujson.dumps(ast_size))
    # with open(os.path.join(save_dir, 'ast_avg_size.dict'), 'w') as writer:
    #     writer.write(ujson.dumps(ast_avg_size))
    #
    # _, path_size, path_avg_size, = data_info(key='path')
    # with open(os.path.join(save_dir, 'path_size.dict'), 'w') as writer:
    #     writer.write(ujson.dumps(path_size))
    # with open(os.path.join(save_dir, 'path_avg_size.dict'), 'w') as writer:
    #     writer.write(ujson.dumps(path_avg_size))
    #
    # _, comment_size, comment_avg_size, = data_info(key='comment')
    # with open(os.path.join(save_dir, 'comment_size.dict'), 'w') as writer:
    #     writer.write(ujson.dumps(comment_size))
    # with open(os.path.join(save_dir, 'comment_avg_size.dict'), 'w') as writer:
    #     writer.write(ujson.dumps(comment_avg_size))


def main():
    expore_dataset()


if __name__ == '__main__':
    main()
