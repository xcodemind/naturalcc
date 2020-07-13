# -*- coding: utf-8 -*-

import json
from ncc.data.indexed_dataset import (
    MMapIndexedDataset,
)
from ncc.data.dictionary import Dictionary


def get_bin_data(file, dict_file):
    bin_dict = Dictionary.load_json(dict_file)
    dataset = MMapIndexedDataset(file)
    bin_data = [bin_dict.string(line) for line in dataset]
    return bin_data


def get_raw_data(file, dict_file):
    raw_dict = Dictionary.load_json(dict_file)
    with open(file, 'r') as reader:
        raw_data = [raw_dict.string([raw_dict.index(token) for token in json.loads(line)]) for line in reader]
    return raw_data


bin_data = get_bin_data(
    file='/home/yang/.ncc/py150/trav_trans_plus/data-mmap/train.ast_trav_df',
    dict_file='/home/yang/.ncc/py150/trav_trans_plus/data-mmap/dict.ast_trav_df.json',
)

raw_data = get_raw_data(
    file='/home/yang/.ncc/py150/trav_trans_plus/data-raw/train.ast_trav_df',
    dict_file='/home/yang/.ncc/py150/trav_trans_plus/data-raw/dict.ast_trav_df.json',
)

assert bin_data == raw_data

bin_data = get_bin_data(
    file='/home/yang/.ncc/py150/trav_trans_plus/data-mmap/test.ast_trav_df',
    dict_file='/home/yang/.ncc/py150/trav_trans_plus/data-mmap/dict.ast_trav_df.json',
)

raw_data = get_raw_data(
    file='/home/yang/.ncc/py150/trav_trans_plus/data-raw/test.ast_trav_df',
    dict_file='/home/yang/.ncc/py150/trav_trans_plus/data-raw/dict.ast_trav_df.json',
)

assert bin_data == raw_data
