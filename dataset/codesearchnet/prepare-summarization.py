# -*- coding: utf-8 -*-
import os
from dataset.codesearchnet.utils import util
from dataset.codesearchnet.utils import util_path
from dataset.codesearchnet.utils import util_ast
from dataset.codesearchnet.utils.util_ast import CodeParser
from dataset.codesearchnet.utils import constants
from typing import Dict, Tuple, List
import gzip
import random
import json, ujson
from copy import deepcopy
import numpy as np
import glob
from ncc import LOGGER
import argparse
from multiprocessing import Pool, cpu_count

random.seed(666)


# raw_dir: codesearchnet的raw data
# clean_dir: clean后我们的"raw data"
# TODO: 现在的实现，里面有很多for lang，for mode，我觉得可以抽到外面去，给定一个lang，和一个mode
def flatten_raw_data(mpool: Pool,
                     raw_dir: str, clean_dir: str,
                     so_file: str, lng: str, modes: List[str]) -> None:
    def raw_data_len(src_filename: str) -> int:
        raw_data = list(util.load_jsonl_gz(src_filename))
        return len(raw_data)

    def flatten(raw_filename: str, dst_filenames: Dict, pop_keys: List[str], start_ind: int,
                so_file: str, lng: str) -> int:
        code_parser = CodeParser(so_file, lng)
        reader = gzip.GzipFile(raw_filename, 'r')
        writers = {
            key: open(dst_filename, 'w')
            for key, dst_filename in dst_filenames.items()
        }
        MAX_SUB_TOKEN_LEN = 0

        data_line = reader.readline().strip()
        while len(data_line) > 0:
            data_line = json.loads(data_line)
            data_line['raw_ast'] = code_parser.parse_raw_ast(data_line['code'])

            for pop_key in pop_keys:
                data_line.pop(pop_key)
            data_line['index'] = start_ind

            max_sub_token_len = 0
            # get max length of a tree nodes' token list
            for _, node in data_line['raw_ast'].items():
                if len(node['children']) == 1:
                    max_sub_token_len = max(max_sub_token_len, len(util.split_identifier(node['children'][0])))
            MAX_SUB_TOKEN_LEN = max(MAX_SUB_TOKEN_LEN, max_sub_token_len)

            for key, entry in data_line.items():
                writers[key].write(ujson.dumps(entry) + '\n')

            start_ind += 1
            data_line = reader.readline().strip()
        return MAX_SUB_TOKEN_LEN

    # get raw filenames
    raw_filenames = {
        mode: util.load_raw_filenames(
            '{}/{}/final/jsonl/{}/*.jsonl.gz'.format(raw_dir, lng, mode),
            sort_func=util.raw_file_index,
            debug=True,
        )
        for mode in modes
    }

    # read raw files and add "index" for case-study
    lengths = {}
    for mode in modes:
        tmp = [mpool.apply_async(raw_data_len, (filename,)) for filename in raw_filenames[mode]]
        lengths[mode] = [res.get() for res in tmp]
    start_indices = {mode: [0] + np.cumsum(lengths[mode]).tolist()[:-1] for mode in modes}

    params = []
    for mode, raw_files in raw_filenames.items():
        for ind, raw_fl in enumerate(raw_files):
            dst_files = {}
            dst_filename = raw_fl.split('/')[-1].replace('.jsonl.gz', '.txt')
            for key in constants.SAVE_KEYS + ['raw_ast', 'index', ]:  # add "raw_ast"
                dst_dir = os.path.join(clean_dir, lng, key, mode, )
                os.makedirs(dst_dir, exist_ok=True)
                dst_files[key] = os.path.join(dst_dir, dst_filename, )
            params.append([raw_fl, dst_files, constants.POP_KEYS, start_indices[mode][ind], \
                           so_file, lng])
    results = [mpool.apply_async(flatten, (param,)) for param in params]
    max_sub_token_lens = [res.get() for res in results]
    max_sub_token_len = max(max_sub_token_lens)
    return max_sub_token_len


# TODO: 给定lang，一次只处理一个lang, 一个mode
# TODO: split raw_ast into ast/path/sbt抽成另一个函数
# TODO: parse_ast_modalities只需要抽出ast即可，path， sbt等单独抽成其他函数
# TODO: save信息页抽出来成一个函数
def parse_ast_modalities(mpool: Pool, clean_dir: str, lng: str, MAX_SUB_TOKEN_LEN: int, ) -> None:
    src_code_files = sorted(glob.glob(os.path.join(clean_dir, lng, 'raw_ast', '*', '*.json.gz')))
    new_modalities = ['path', 'sbt', 'sbtao', 'ast', ]
    dst_dir = {}
    for modal in new_modalities:
        dst_dir[modal] = os.path.join(clean_dir, lng, 'modal', )
        os.makedirs(dst_dir[modal], exist_ok=True)

    def parse_new_modalities(code_file: str, ) -> None:
        reader = open(code_file, 'r')
        dst_files = {modal: os.path.join(dst_dir[modal], code_file.split('/')[-1]) for modal in new_modalities}
        writers = {
            modal: open(dst_files[modal], 'w')
            for modal in new_modalities
        }
        MAX_SUB_TOKEN_LEN = 0

        data_line = reader.readline().strip()
        while len(data_line) > 0:
            raw_ast = json.loads(data_line)

            if 'path' in writers:
                path = util_path.ast_to_path(deepcopy(raw_ast))
                writers['path'].write(ujson.dumps(path) + '\n')

            if 'sbt' in writers:
                # sbt
                padded_raw_ast = util_ast.pad_leaf_node(deepcopy(raw_ast), MAX_SUB_TOKEN_LEN)
                sbt = util_ast.parse_deepcom(padded_raw_ast, util_ast.build_sbt_tree, to_lower=True, )
                writers['sbt'].write(ujson.dumps(sbt) + '\n')

            if 'sbtao' in writers:
                # sbt
                padded_raw_ast = util_ast.pad_leaf_node(deepcopy(raw_ast), MAX_SUB_TOKEN_LEN)
                sbt = util_ast.parse_deepcom(padded_raw_ast, util_ast.build_sbtao_tree, to_lower=True, )
                writers['sbtao'].write(ujson.dumps(sbt) + '\n')

            data_line = reader.readline().strip()

    results = [mpool.apply_async(parse_new_modalities, (code_file,)) for code_file in src_code_files]
    results = [res.get() for res in results]


def ast2path(xx):
    pass


def ast2sbt(xx):
    pass


def main():
    parser = argparse.ArgumentParser()
    # Before creating the true parser, we need to import optional user module
    # in order to eagerly import custom tasks, optimizers, architectures, etc.
    parser.add_argument('--raw_dir', default='/data/wanyao/ghproj_d/naturalcodev3/codesearchnet/raw')
    parser.add_argument('--clean_dir', default='/data/wanyao/ghproj_d/naturalcodev3/codesearchnet/clean')
    args_ = parser.parse_args()

    os.makedirs(args_.clean_dir, exist_ok=True)
    ################################################################
    # use all cores
    ################################################################
    mpool = Pool(processes=1)  # build a multi-processing pool
    lng = constants.LANGUAGES
    modes = constants.MODES
    so_file = ''  # your tree_sitter so file

    # 1. flatten
    max_sub_token_len = flatten_raw_data(mpool, args_.raw_dir, args_.clean_dir,
                                         so_file, lng, modes, )

    # 2. parse ast
    parse_ast_modalities(mpool, args_.clean_dir, lng, max_sub_token_len)

    # 3. ast2path

    #  4. ast2sbt

    # 到此结束，dict构造我会放到preprocess.py里面去
    # from dataset.parse_key.dicts import main as dict_main
    # dict_main(
    #     dataset_dir=clean_dir,
    #     KEYS=None,
    #     xlang=True,
    # )


if __name__ == '__main__':
    main()
