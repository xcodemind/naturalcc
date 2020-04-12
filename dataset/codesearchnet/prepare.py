# -*- coding: utf-8 -*-
import os
from dataset.codesearchnet.utils import util
from dataset.codesearchnet.utils import util_path
from dataset.codesearchnet.utils.util_ast import CodeParser
from dataset.codesearchnet.utils import constants
from typing import Dict, Tuple, List
import gzip
import random
import json, ujson
from copy import deepcopy
import numpy as np
from ncc import LOGGER
from joblib import Parallel, delayed
import argparse

random.seed(666)

# raw_dir: codesearchnet的raw data
# clean_dir: clean后我们的"raw data"
# TODO: 现在的实现，里面有很多for lang，for mode，我觉得可以抽到外面去，给定一个lang，和一个mode
# def flatten_raw_data(paralleler: Parallel, raw_dir: str, clean_dir: str, lang: str, mode: str) -> None:

def flatten_raw_data(paralleler: Parallel, raw_dir: str, clean_dir: str, ) -> None:
    def raw_data_len(src_filename: str) -> int:
        raw_data = list(util.load_jsonl_gz(src_filename))
        return len(raw_data)

    def parse_flatten(raw_filename: str, pop_keys: List[str], start_ind: int,
                      so_file: str, language: str, dst_filenames: Dict, ) -> Tuple:
        reader = gzip.GzipFile(raw_filename, 'r')
        writers = {
            key: open(dst_filename, 'w')
            for key, dst_filename in dst_filenames.items()
        }
        code_parser = CodeParser(so_file, language)
        MAX_SUB_TOKEN_LEN = 0

        data_line = reader.readline().strip()
        while len(data_line) > 0:
            data_line = json.loads(data_line)

            for pop_key in pop_keys:
                data_line.pop(pop_key)
            data_line['index'] = start_ind

            ################################################################
            # split code into tok
            # split docstring into comment
            # split func_name into method
            # parse raw_ast
            ################################################################
            data_line['tok'], error, = code_parser.parse_tok(data_line['code_tokens'])
            if data_line['tok'] is None:
                writers['bad_cases'].write('parse code_tokens error({})\n'.format(error))
                writers['bad_cases'].write(ujson.dumps(data_line) + '\n\n')

                start_ind += 1
                data_line = reader.readline().strip()
                continue

            ################################################################
            data_line['comment'], error = code_parser.parse_comment(data_line['docstring'],
                                                                    data_line['docstring_tokens'])
            if data_line['comment'] is None:
                writers['bad_cases'].write('parse docstring_tokens error({})\n'.format(error))
                writers['bad_cases'].write(ujson.dumps(data_line) + '\n\n')

                start_ind += 1
                data_line = reader.readline().strip()
                continue

            if len(error) > 0:
                writers['bad_cases'].write('parse docstring_tokens error({})\n'.format(error))
                writers['bad_cases'].write(ujson.dumps(data_line) + '\n\n')

            ################################################################
            raw_ast = code_parser.parse_raw_ast(data_line['code'])
            data_line['raw_ast'] = raw_ast
            max_sub_token_len = 0
            # get max length of a tree nodes' token list
            for _, node in data_line['raw_ast'].items():
                if len(node['children']) == 1:
                    max_sub_token_len = max(max_sub_token_len, len(util.split_identifier(node['children'][0])))
            MAX_SUB_TOKEN_LEN = max(MAX_SUB_TOKEN_LEN, max_sub_token_len)

            data_line['method'] = code_parser.parse_method(data_line['func_name'])

            for key, entry in data_line.items():
                writers[key].write(ujson.dumps(entry) + '\n')

            # # for debug
            # break

            start_ind += 1
            data_line = reader.readline().strip()
        return language, MAX_SUB_TOKEN_LEN,

    ################################################################
    # get each files' start index
    # raw={
    # 'code':XXX, 'docstring':xx, ...
    # }
    # => code, docstring,
    ################################################################
    raw_filenames = {
        lng: {
            mode: util.load_raw_filenames(
                '{}/{}/final/jsonl/{}/*.jsonl.gz'. \
                    format(raw_dir, lng, mode),
                sort_func=util.raw_file_index,
                # debug=True,
            )
            for mode in constants.MODES
        }
        for lng in constants.LANGUAGES
    }
    # LOGGER.debug(raw_filenames)

    # read raw file first to add "index" in later datasets for later case-study
    raw_lens = {
        lng: {
            mode: paralleler(delayed(raw_data_len)(filename) for filename in raw_filenames[lng][mode])
            for mode in constants.MODES
        }
        for lng in constants.LANGUAGES
    }
    # LOGGER.debug(raw_lens)

    raw_start_indices = {
        lng: {
            mode: [0] + np.cumsum(raw_lens[lng][mode]).tolist()[:-1]
            for mode in constants.MODES
        }
        for lng in constants.LANGUAGES
    }
    # LOGGER.debug(raw_start_indices)

    ################################################################
    # read raw file, and save entries into different *.jsonl.gz
    ################################################################

    params = []
    for lng in constants.LANGUAGES:
        for mode, raw_fls in raw_filenames[lng].items():
            for ind, raw_fl in enumerate(raw_fls):
                dst_fls = {}
                dst_flname = raw_fl.split('/')[-1].replace('.jsonl.gz', '.txt')
                for key in constants.SAVE_KEYS + ['index', 'raw_ast', 'tok', 'comment', 'method', 'bad_cases']:
                    dst_dir = os.path.join(clean_dir, lng, key, )
                    os.makedirs(dst_dir, exist_ok=True)
                    dst_fls[key] = os.path.join(dst_dir, dst_flname, )
                params.append([raw_fl, constants.POP_KEYS, raw_start_indices[lng][mode][ind], constants.SO_FILE, lng, dst_fls, ])
    # LOGGER.debug(params)
    lng_lens = paralleler(delayed(parse_flatten)(*param) for param in params)
    LOGGER.info(lng_lens)

    ################################################################
    # write max sub-token len into info.txt
    ################################################################
    lngs_info = {lng: 0 for lng in constants.LANGUAGES}
    for lng, max_len in lng_lens:
        lngs_info[lng] = max(lngs_info[lng], max_len)
    LOGGER.debug(lngs_info)

    info_file = os.path.join(clean_dir, 'info.txt')
    with open(info_file, 'w') as writer:
        writer.write(ujson.dumps(lngs_info))
    LOGGER.info('write MAX_SUB_TOKEN_LEN in {}'.format(info_file))

# TODO: 给定lang，一次只处理一个lang, 一个mode
# TODO: split raw_ast into ast/path/sbt/sbt2抽成另一个函数
# TODO: parse_ast_modalities只需要抽出ast即可，path， sbt等单独抽成其他函数
# TODO: save信息页抽出来成一个函数
# def parse_ast_modalities(paralleler: Parallel, lang: str, mode: str,) -> None:
def parse_ast_modalities(paralleler: Parallel, clean_dir: str, ) -> None:
    def parse_new_ast_modalities(raw_ast_filename: str, MAX_SUB_TOKEN_LEN: int, new_modalities_filanems: Dict, ):
        reader = open(raw_ast_filename, 'r')
        writers = {
            key: open(filename, 'w')
            for key, filename in new_modalities_filanems.items()
        }

        raw_ast = reader.readline().strip()
        while len(raw_ast) > 0:
            raw_ast = ujson.loads(raw_ast)

            if 'path' in new_modalities_filanems:
                # path
                path = util.ast_to_path(deepcopy(raw_ast))
                # in raw dataset, save all paths
                # if len(path) > PATH_K:
                #     sampled_ind = random.sample(range(len(path)), PATH_K)
                # elif len(path) == PATH_K:
                #     sampled_ind = list(range(len(path)))
                # else:
                #     sampled_ind = list(range(len(path)))
                #     sampled_ind = sampled_ind * (PATH_K // len(path))
                #     appended_ind = [random.randint(0, len(path) - 1) for _ in range(PATH_K % len(path))]
                #     sampled_ind.extend(appended_ind)
                # random.shuffle(sampled_ind)
                # path = [path[ind] for ind in sampled_ind]
                writers['path'].write(ujson.dumps(path) + '\n')

            if 'sbt' in new_modalities_filanems:
                # sbt
                padded_raw_ast = util.pad_leaf_node(deepcopy(raw_ast), MAX_SUB_TOKEN_LEN)
                sbt = util.parse_deepcom(padded_raw_ast, util_path.build_sbt_tree, to_lower=True, )
                writers['sbt'].write(ujson.dumps(sbt) + '\n')

            if 'sbtao' in new_modalities_filanems:
                # sbt
                padded_raw_ast = util.pad_leaf_node(deepcopy(raw_ast), MAX_SUB_TOKEN_LEN)
                sbt = util.parse_deepcom(padded_raw_ast, util_path.build_sbtao_tree, to_lower=True, )
                writers['sbtao'].write(ujson.dumps(sbt) + '\n')

            if 'sbt2' in new_modalities_filanems:
                # sbt2
                sbt2 = util.parse_deepcom(padded_raw_ast, util_path.build_sbt2_tree, to_lower=True, )
                writers['sbt2'].write(ujson.dumps(sbt2) + '\n')

            if 'ast' in new_modalities_filanems:
                # ast
                ast = util.pad_leaf_node(util_path.parse_base(raw_ast), MAX_SUB_TOKEN_LEN)
                writers['ast'].write(ujson.dumps(ast) + '\n')

            raw_ast = reader.readline().strip()

    ################################################################
    # split raw_ast into ast/path/sbt/sbt2
    ################################################################
    info_file = os.path.join(clean_dir, 'info.txt')
    with open(info_file, 'r') as reader:
        lngs_info = ujson.loads(reader.read().strip())

    params = []
    for lng in constants.LANGUAGES:
        filenames = os.listdir(os.path.join(clean_dir, lng, 'raw_ast'))
        for filename in filenames:
            dst_filenames = {}
            for key in ['ast', 'path', 'sbt', 'sbtao', 'sbt2', ]:
                dst_dir = os.path.join(clean_dir, lng, key, )
                os.makedirs(dst_dir, exist_ok=True)
                dst_filenames[key] = os.path.join(dst_dir, filename)
            params.append([os.path.join(clean_dir, lng, 'raw_ast', filename), lngs_info[lng], dst_filenames])
    LOGGER.debug(params)
    paralleler(delayed(parse_new_ast_modalities)(*param) for param in params)

    '''
    remove path/ border.dict center.dict

    rm -fr */*/path/

    rm */*/*.border.dict
    rm */*/*.center.dict

    rm */*.border.dict
    rm */*.center.dict

    python -u ./dataset/parse_key/main.py > key_100.log 2>&1 &
    '''


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
    from multiprocessing import cpu_count
    # paralleler = Parallel(n_jobs=cpu_count())  # build a multi-processing pool
    paralleler = Parallel(n_jobs=1)  # build a multi-processing pool
    # 1. flatten
    flatten_raw_data(paralleler, args_.raw_dir, args_.clean_dir)

    # 2. parse ast
    # parse_ast_modalities(paralleler, clean_dir)

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
