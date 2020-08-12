# -*- coding: utf-8 -*-

import os
import ujson
import random
import argparse
import itertools
from multiprocessing import Pool, cpu_count
from ncc.utils.mp_ppool import PPool, cpu_count
from dataset.csn import PATH_NUM, MAX_SUB_TOKEN_LEN, MODES

try:
    from dataset.csn import (
        LANGUAGES, MODES,
        RAW_DATA_DIR, LIBS_DIR, FLATTEN_DIR, REFINE_DIR,
        LOGGER,
    )
    from dataset.csn.parser._parser import CodeParser
    from dataset.csn.utils import (util, util_ast, util_path, util_traversal)
except ImportError:
    from . import (
        LANGUAGES, MODES,
        RAW_DATA_DIR, LIBS_DIR, FLATTEN_DIR, REFINE_DIR,
        LOGGER,
    )
    from dataset.csn.parser._parser import CodeParser
    from .utils import (util, util_ast, util_path, util_traversal)


def find_offsets(filename, num_chunks):
    with open(filename, "r", encoding="utf-8") as f:
        size = os.fstat(f.fileno()).st_size
        chunk_size = size // num_chunks
        offsets = [0 for _ in range(num_chunks + 1)]
        for i in range(1, num_chunks):
            f.seek(chunk_size * i)
            safe_readline(f)
            offsets[i] = f.tell()
        return offsets


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


if __name__ == '__main__':
    """
    This script is to generate new attributes of code snippet.
    """
    parser = argparse.ArgumentParser(description="Download CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--language", "-l", default=LANGUAGES, type=str, nargs='+', help="languages constain [{}]".format(LANGUAGES),
    )
    parser.add_argument(
        "--flatten_dir", "-f", default=FLATTEN_DIR, type=str, help="data directory of flatten attributes",
    )
    parser.add_argument(
        "--refine_dir", "-r", default=REFINE_DIR, type=str, help="refine directory from flatten attributes",
    )
    parser.add_argument(
        "--attrs", "-a",
        default=['code', 'code_tokens', 'docstring', 'docstring_tokens',
                 'raw_ast', 'ast', 'path', 'path.terminals', 'sbt', 'sbtao', 'binary_ast', 'traversal'], type=list,
        help="attrs: raw_ast, ...",
    )
    args = parser.parse_args()
    print(args)

    args.flatten_dir = os.path.expanduser(args.flatten_dir)
    args.refine_dir = os.path.expanduser(args.refine_dir)
    NONE_LINE = 'null\n'


    def filter_fn(attrs):
        LOGGER.info('Filtering {}'.format(attrs))
        readers, writers = {}, {}
        for attr in attrs:
            readers[attr] = open(os.path.join(args.flatten_dir, lang, '{}.{}'.format(mode, attr)), 'r',
                                 encoding='UTF-8')
            dst_dir = os.path.join(args.refine_dir, lang)
            os.makedirs(dst_dir, exist_ok=True)
            writers[attr] = open(os.path.join(dst_dir, '{}.{}'.format(mode, attr)), 'w')

        def _read_attrs_lines():
            data_info = []
            drop = False
            for attr in attrs:
                if attr == 'path.terminals':
                    line = [readers[attr].readline() for _ in range(PATH_NUM * 2)]  # 2 for head and tail of a path
                    if all([hbt == NONE_LINE for hbt in line]):
                        drop = True
                elif attr == 'path':
                    line = [readers[attr].readline() for _ in range(PATH_NUM)]  # body of a path
                    if all([hbt == NONE_LINE for hbt in line]):
                        drop = True
                else:
                    line = readers[attr].readline()
                    if line == NONE_LINE:
                        drop = True
                data_info.append(line)
            return data_info, drop

        end = os.fstat(readers[attrs[0]].fileno()).st_size
        while readers[attrs[0]].tell() < end:
            data_info, drop = _read_attrs_lines()
            if not drop:
                for idx, attr in enumerate(attrs):
                    if attr == 'path.terminals':
                        for line in data_info[idx]:
                            writers[attr].write(line)
                    elif attr == 'path':
                        for line in data_info[idx]:
                            writers[attr].write(line)
                    else:
                        writers[attr].write(data_info[idx])

        for attr, reader in readers.items():
            reader.close()
            writers[attr].close()


    for lang, mode in itertools.product(args.language, MODES):
        filter_fn(args.attrs)
