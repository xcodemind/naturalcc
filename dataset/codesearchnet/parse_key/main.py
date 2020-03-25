# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.path.abspath('.'))

from dataset.utils.util_ast import *
from ncc.utils.constants import *

import random

random.seed(666)


def flatten_raw_data(paralleler: Parallel, raw_dir: str, ) -> None:
    def raw_data_len(src_filename: str) -> int:
        raw_data = list(load_jsonl_gz(src_filename))
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
                    max_sub_token_len = max(max_sub_token_len, len(split_identifier(node['children'][0])))
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
            mode: load_raw_filenames(
                '/data/wanyao/ghproj_d/CodeSearchNet/data/{}/final/jsonl/{}/*.jsonl.gz'. \
                    format(lng, mode),
                sort_func=raw_file_index,
                # debug=True,
            )
            for mode in MODES
        }
        for lng in LANGUAGES
    }
    # LOGGER.debug(raw_filenames)

    # read raw file first to add "index" in later datasets for later case-study
    raw_lens = {
        lng: {
            mode: paralleler(delayed(raw_data_len)(filename) for filename in raw_filenames[lng][mode])
            for mode in MODES
        }
        for lng in LANGUAGES
    }
    # LOGGER.debug(raw_lens)

    raw_start_indices = {
        lng: {
            mode: [0] + np.cumsum(raw_lens[lng][mode]).tolist()[:-1]
            for mode in MODES
        }
        for lng in LANGUAGES
    }
    # LOGGER.debug(raw_start_indices)

    ################################################################
    # read raw file, and save entries into different *.jsonl.gz
    ################################################################

    params = []
    for lng in LANGUAGES:
        for mode, raw_fls in raw_filenames[lng].items():
            for ind, raw_fl in enumerate(raw_fls):
                dst_fls = {}
                dst_flname = raw_fl.split('/')[-1].replace('.jsonl.gz', '.txt')
                for key in SAVE_KEYS + ['index', 'raw_ast', 'tok', 'comment', 'method', 'bad_cases']:
                    dst_dir = os.path.join(raw_dir, lng, key, )
                    os.makedirs(dst_dir, exist_ok=True)
                    dst_fls[key] = os.path.join(dst_dir, dst_flname, )
                params.append([raw_fl, POP_KEYS, raw_start_indices[lng][mode][ind], SO_FILE, lng, dst_fls, ])
    # LOGGER.debug(params)
    lng_lens = paralleler(delayed(parse_flatten)(*param) for param in params)
    LOGGER.info(lng_lens)

    ################################################################
    # write max sub-token len into info.txt
    ################################################################
    lngs_info = {lng: 0 for lng in LANGUAGES}
    for lng, max_len in lng_lens:
        lngs_info[lng] = max(lngs_info[lng], max_len)
    LOGGER.debug(lngs_info)

    info_file = os.path.join(raw_dir, 'info.txt')
    with open(info_file, 'w') as writer:
        writer.write(ujson.dumps(lngs_info))
    LOGGER.info('write MAX_SUB_TOKEN_LEN in {}'.format(info_file))


def parse_ast_modalities(paralleler: Parallel, raw_dir: str, ) -> None:
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
                path = ast_to_path(deepcopy(raw_ast))
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
                padded_raw_ast = pad_leaf_node(deepcopy(raw_ast), MAX_SUB_TOKEN_LEN)
                sbt = parse_deepcom(padded_raw_ast, build_sbt_tree, to_lower=True, )
                writers['sbt'].write(ujson.dumps(sbt) + '\n')

            if 'sbtao' in new_modalities_filanems:
                # sbt
                padded_raw_ast = pad_leaf_node(deepcopy(raw_ast), MAX_SUB_TOKEN_LEN)
                sbt = parse_deepcom(padded_raw_ast, build_sbtao_tree, to_lower=True, )
                writers['sbtao'].write(ujson.dumps(sbt) + '\n')

            if 'sbt2' in new_modalities_filanems:
                # sbt2
                sbt2 = parse_deepcom(padded_raw_ast, build_sbt2_tree, to_lower=True, )
                writers['sbt2'].write(ujson.dumps(sbt2) + '\n')

            if 'ast' in new_modalities_filanems:
                # ast
                ast = pad_leaf_node(parse_base(raw_ast), MAX_SUB_TOKEN_LEN)
                writers['ast'].write(ujson.dumps(ast) + '\n')

            raw_ast = reader.readline().strip()

    ################################################################
    # split raw_ast into ast/path/sbt/sbt2
    ################################################################
    info_file = os.path.join(raw_dir, 'info.txt')
    with open(info_file, 'r') as reader:
        lngs_info = ujson.loads(reader.read().strip())

    params = []
    for lng in LANGUAGES:
        filenames = os.listdir(os.path.join(raw_dir, lng, 'raw_ast'))
        for filename in filenames:
            dst_filenames = {}
            for key in ['ast', 'path', 'sbt', 'sbtao', 'sbt2', ]:
                dst_dir = os.path.join(raw_dir, lng, key, )
                os.makedirs(dst_dir, exist_ok=True)
                dst_filenames[key] = os.path.join(dst_dir, filename)
            params.append([os.path.join(raw_dir, lng, 'raw_ast', filename), lngs_info[lng], dst_filenames])
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


def main():
    raw_dir = RAW_DATASET_DIR
    os.makedirs(raw_dir, exist_ok=True)
    ################################################################
    # use all cores
    ################################################################
    from multiprocessing import cpu_count
    # paralleler = Parallel(n_jobs=cpu_count())  # build a multi-processing pool
    paralleler = Parallel(n_jobs=2)  # build a multi-processing pool
    # flatten_raw_data(paralleler, raw_dir)
    parse_ast_modalities(paralleler, raw_dir)

    from dataset.parse_key.dicts import main as dict_main
    dict_main(
        dataset_dir=raw_dir,
        KEYS=None,
        xlang=True,
    )


if __name__ == '__main__':
    main()
