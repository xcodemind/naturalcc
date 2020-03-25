# -*- coding: utf-8 -*-

import os
import sys

sys.path.append('.')

from typing import *

from dataset.utils.util import *
from dataset.utils.util_ast import *
from dataset.utils.util_dict import *
from ncc.data.dict import Dict as _Dict


def merge_counters(counter_list: List[Counter]) -> Dict:
    dict_list = [{
        key: vaue
        for key, vaue in counter.items()
    } for counter in counter_list if counter is not None]
    token_dict = merge_freqency(dict_list)
    return token_dict


def list_to_dict(dst_dir: str, dict_filename: str, min_freq=2):
    train_files = [file for file in glob('{}/*'.format(dst_dir)) if 'train' in file.split('/')[-1]]
    train_counters = []
    list_buffer = []
    for file in train_files:
        with open(file, 'r') as reader:
            line = reader.readline().strip()
            while len(line) > 0:
                line = ujson.loads(line)
                list_buffer.append(line)

                if len(list_buffer) >= 5000:
                    train_counters.append(Counter(itertools.chain(*list_buffer)))
                    list_buffer = []

                line = reader.readline().strip()
    if len(list_buffer) > 0:
        train_counters.append(Counter(itertools.chain(*list_buffer)))
        list_buffer = []

    token_dict = merge_counters(train_counters)
    tokens_dict = {key: freq for key, freq in token_dict.items() if freq > min_freq}
    sorted_tokens = sort_by_freq(tokens_dict)
    dump_dict(sorted_tokens, dict_filename)


def tree_to_dict(dst_dir: str, dict_filename: str, min_freq=2):
    train_files = [file for file in glob('{}/*'.format(dst_dir)) if 'train' in file.split('/')[-1]]
    train_counters = []
    list_buffer = []
    for file in train_files:
        with open(file, 'r') as reader:
            line = reader.readline().strip()
            while len(line) > 0:
                line = ujson.loads(line)

                leaf_node_tokens = []
                for node_inf, node_info in line.items():
                    if type(node_info['children'][-1]) == list:
                        for token in node_info['children'][-1]:
                            if token != PAD_WORD:
                                leaf_node_tokens.append(token)
                            else:
                                break
                list_buffer.append(leaf_node_tokens)

                if len(list_buffer) >= 5000:
                    train_counters.append(Counter(itertools.chain(*list_buffer)))
                    list_buffer = []
                line = reader.readline().strip()
    if len(list_buffer) > 0:
        train_counters.append(Counter(itertools.chain(*list_buffer)))
        list_buffer = []

    token_dict = merge_counters(train_counters)
    tokens_dict = {key: freq for key, freq in token_dict.items() if freq > min_freq}
    sorted_tokens = sort_by_freq(tokens_dict)
    dump_dict(sorted_tokens, dict_filename)


def path_to_dict(dst_dir: str, border_dict_filename: str, center_dict_filename: str, min_freq=2):
    train_files = [file for file in glob('{}/*'.format(dst_dir)) if 'train' in file.split('/')[-1]]
    border_counters, center_counters = [], []
    border_list_buffer, center_list_buffer = [], []
    for file in train_files:
        with open(file, 'r') as reader:
            line = reader.readline().strip()
            while len(line) > 0:
                line = ujson.loads(line)

                for path in line:
                    head, center, tail = path
                    border_list_buffer.append(head + tail)
                    center_list_buffer.append(center)

                if len(border_list_buffer) >= 5000 or len(center_list_buffer) > 5000:
                    border_counters.append(Counter(itertools.chain(*border_list_buffer)))
                    center_counters.append(Counter(itertools.chain(*center_list_buffer)))
                    border_list_buffer, center_list_buffer = [], []
                line = reader.readline().strip()
    if len(border_list_buffer) >= 0 or len(center_list_buffer) > 0:
        border_counters.append(Counter(itertools.chain(*border_list_buffer)))
        center_counters.append(Counter(itertools.chain(*center_list_buffer)))
        border_list_buffer, center_list_buffer = [], []

    border_dict = merge_counters(border_counters)
    border_dict = {key: freq for key, freq in border_dict.items() if freq > min_freq}
    sorted_border_tokens = sort_by_freq(border_dict)
    dump_dict(sorted_border_tokens, border_dict_filename)

    center_dict = merge_counters(center_counters)
    center_dict = {key: freq for key, freq in center_dict.items() if freq > min_freq}
    sorted_center_tokens = sort_by_freq(center_dict)
    dump_dict(sorted_center_tokens, center_dict_filename)


def load_token_dict(dst_filename: str) -> Dict:
    token_freq_dict = {}
    with open(dst_filename, 'r') as reader:
        data = ujson.loads(reader.read())
        for token, freq in data:
            token_freq_dict[token] = freq
    return token_freq_dict


def dump_xlang_token_dict(dict_filename: str, dict_pairs: List[Dict], ) -> None:
    if os.path.exists(dict_filename):
        return
    else:
        LOGGER.info('write {}'.format(dict_filename))
        # print('write {}'.format(dict_filename))
    token_dict = merge_freqency(dict_pairs)
    token_dict = {key: freq for key, freq in token_dict.items()}
    sorted_token_dict = sort_by_freq(token_dict)
    dump_dict(sorted_token_dict, dict_filename)


def main(dataset_dir: str, KEYS: List, xlang=True, ):
    ################################################################
    # use all cores
    ################################################################
    from multiprocessing import cpu_count
    paralleler = Parallel(n_jobs=cpu_count())  # build a multi-processing pool

    LANGUAGES = ['python', 'java', 'go', 'php', 'ruby', 'javascript']
    if KEYS is None:
        KEYS = [
            'tok', 'code_tokens', 'sbt', 'sbtao', 'sbt2', 'comment', 'docstring_tokens', 'method',
            'ast',  # ast
            'border', 'center',  # path
        ]

    ################################################################
    # unilang dicts
    # generate dicts
    # tok/code_tokens/sbt/sb2/comment/docstring_tokens/method -> list
    # ast -> dict
    # path(border, center) -> 2 lists
    # 1) python dict, token_freq>2, -> save -> token<50000

    # freq>2
    # multi-lingusitc [python,ruby] - sort([python,ruby])-> ([python,ruby])[:50000]
    ################################################################

    if 'ast' in KEYS:
        params = []
        for lng in LANGUAGES:
            dst_dir = os.path.join(dataset_dir, lng, 'ast')
            dict_filename = os.path.join(dataset_dir, lng, '{}.ast.dict'.format(lng))
            params.append([dst_dir, dict_filename])
        # LOGGER.debug(params)
        # tree_to_dict(params[0])
        paralleler(delayed(tree_to_dict)(*param) for param in params)
        KEYS -= ['ast']

    if 'path' in KEYS:
        params = []
        for lng in LANGUAGES:
            dst_dir = os.path.join(dataset_dir, lng, 'path')
            border_dict_filename = os.path.join(dataset_dir, lng, '{}.border.dict'.format(lng, ))
            center_dict_filename = os.path.join(dataset_dir, lng, '{}.center.dict'.format(lng, ))
            params.append([dst_dir, border_dict_filename, center_dict_filename])
        # LOGGER.debug(params)
        # path_to_dict(params[0])
        paralleler(delayed(path_to_dict)(*param) for param in params)
        KEYS -= ['path']

    params = []
    for lng in LANGUAGES:
        for key in KEYS:
            dst_dir = os.path.join(dataset_dir, lng, key)
            dict_filename = os.path.join(dataset_dir, lng, '{}.{}.dict'.format(lng, key))
            params.append([dst_dir, dict_filename])
    # LOGGER.debug(params)
    # list_to_dict(*params[0])
    paralleler(delayed(list_to_dict)(*param) for param in params)

    ################################################################
    # xlang dicts
    ################################################################
    if xlang:
        lng_token_dicts = {
            lng: {
                key: load_token_dict(os.path.join(dataset_dir, lng, '{}.{}.dict'.format(lng, key)))
                for key in KEYS
            }
            for lng in LANGUAGES
        }

        LOGGER.info('shared dicts')
        params = []
        for num in range(2, len(LANGUAGES) + 1):
            for lng_pairs in itertools.permutations(LANGUAGES, num):
                for key in KEYS:
                    lng_pairs = sorted(lng_pairs)
                    dict_filename = os.path.join(dataset_dir, '{}.{}.dict'.format('_'.join(lng_pairs), key))
                    dict_pairs = [lng_token_dicts[lng][key] for lng in lng_pairs]
                    params.append([dict_filename, dict_pairs, ])
        paralleler(delayed(dump_xlang_token_dict)(*param) for param in params)

# if __name__ == '__main__':
#     main(dataset_dir=KEY_DST_DIR)
