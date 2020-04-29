# -*- coding: utf-8 -*-

from typing import List, Dict, Tuple

import os
import numpy as np
import gzip
import ujson
import json
from multiprocessing import cpu_count, Pool
import random

random.seed(666)

from dataset.codesearchnet.utils import util
from dataset.codesearchnet.utils import util_ast
from dataset.codesearchnet.utils import constants



def flatten_raw_data(mpool: Pool, raw_dir: str, ) -> None:
    ################################################################
    # get each files' start index
    # raw={
    # 'code':XXX, 'docstring':xx, ...
    # }
    # => code, docstring,
    ################################################################
    # LANGUAGES = ['javascript']
    raw_filenames = {
        lng: {
            mode: util.load_raw_filenames(
                '/data/wanyao/ghproj_d/CodeSearchNet/data/{}/final/jsonl/{}/*.jsonl.gz'. \
                    format(lng, mode),
                sort_func=util.raw_file_index,
                # debug=True,
            )
            for mode in constants.MODES
        }
        for lng in constants.LANGUAGES
    }
    print(raw_filenames)

    # read raw file first to add "index" in later datasets for later case-study
    raw_lens = {}
    for lng in constants.LANGUAGES:
        raw_lens[lng] = {}
        for mode in constants.MODES:
            result = [mpool.apply_async(raw_data_len, (filename,)) for filename in raw_filenames[lng][mode]]
            raw_lens[lng][mode] = [res.get() for res in result]

    raw_start_indices = {
        lng: {
            mode: [0] + np.cumsum(raw_lens[lng][mode]).tolist()[:-1]
            for mode in constants.MODES
        }
        for lng in constants.LANGUAGES
    }
    print(raw_start_indices)

    ################################################################
    # read raw file, and save entries into different *.jsonl.gz
    ################################################################
    lng_lens = []
    for lng in constants.LANGUAGES:
        params = []
        for mode, raw_fls in raw_filenames[lng].items():
            for ind, raw_fl in enumerate(raw_fls):
                dst_fls = {}
                dst_flname = raw_fl.split('/')[-1].replace('.jsonl.gz', '.txt')
                SO_FILE = '/data/wanyao/yang/ghproj_d/GitHub/naturalcodev2/dataset/parser_zips/{}.so'.format(lng)
                for key in ['code', 'func_name', 'code_tokens', 'docstring_tokens', 'index', 'raw_ast', 'method', ]:
                    dst_dir = os.path.join(raw_dir, lng, key, )
                    os.makedirs(dst_dir, exist_ok=True)
                    dst_fls[key] = os.path.join(dst_dir, dst_flname, )
                POP_KEYS = ['original_string', 'path', 'repo', 'sha', 'url', 'language', 'partition', 'docstring']
                params.append([raw_fl, POP_KEYS, raw_start_indices[lng][mode][ind], SO_FILE, lng, dst_fls, ])
        result = [mpool.apply_async(parse_flatten, (*param,)) for param in params]
        result = [res.get() for res in result]
        lng_lens.extend(result)

    ################################################################
    # write max sub-token len into info.txt
    ################################################################
    lngs_info = {lng: 0 for lng in constants.LANGUAGES}
    for lng, max_len in lng_lens:
        lngs_info[lng] = max(lngs_info[lng], max_len)

    info_file = os.path.join(raw_dir, 'info.txt')
    with open(info_file, 'w') as writer:
        writer.write(ujson.dumps(lngs_info))
    print('write MAX_SUB_TOKEN_LEN in {}'.format(info_file))


def parse_new_ast_modalities(raw_ast_filename: str, MAX_SUB_TOKEN_LEN: int, new_modalities_filanems: Dict, ):
    reader = open(raw_ast_filename, 'r')
    writers = {
        key: open(filename, 'w')
        for key, filename in new_modalities_filanems.items()
    }

    raw_ast = reader.readline().strip()
    while len(raw_ast) > 0:
        raw_ast = ujson.loads(raw_ast)

        if 'ast' in new_modalities_filanems:
            # ast
            ast = util_ast.pad_leaf_node(util_ast.parse_base(raw_ast), MAX_SUB_TOKEN_LEN)
            writers['ast'].write(ujson.dumps(ast) + '\n')

        raw_ast = reader.readline().strip()


def parse_ast_modalities(mpool: Pool, raw_dir: str, ) -> None:
    ################################################################
    # split raw_ast into ast
    ################################################################
    info_file = os.path.join(raw_dir, 'info.txt')
    with open(info_file, 'r') as reader:
        lngs_info = ujson.loads(reader.read().strip())

    params = []
    for lng in constants.LANGUAGES:
        filenames = os.listdir(os.path.join(raw_dir, lng, 'raw_ast'))
        for filename in filenames:
            dst_filenames = {}
            for key in ['ast', ]:
                dst_dir = os.path.join(raw_dir, lng, key, )
                os.makedirs(dst_dir, exist_ok=True)
                dst_filenames[key] = os.path.join(dst_dir, filename)
            params.append([os.path.join(raw_dir, lng, 'raw_ast', filename), lngs_info[lng], dst_filenames])
    result = [mpool.apply_async(parse_new_ast_modalities, (*param,)) for param in params]
    result = [res.get() for res in result]
