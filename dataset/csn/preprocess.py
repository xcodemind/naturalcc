# -*- coding: utf-8 -*-

import os
import ujson
import argparse
import itertools
from multiprocessing import Pool, cpu_count
from ncc.utils.mp_ppool import PPool, cpu_count

try:
    from dataset.csn import (
        LANGUAGES, MODES,
        RAW_DATA_DIR, LIBS_DIR, FLATTEN_DIR,
        LOGGER,
    )
    from dataset.csn.parser._parser import CodeParser
    from dataset.csn.utils import util
    from dataset.csn.utils import util_ast
    from dataset.csn.utils import util_path
except ImportError:
    from . import (
        LANGUAGES, MODES,
        RAW_DATA_DIR, LIBS_DIR, FLATTEN_DIR,
        LOGGER,
    )
    from dataset.csn.parser._parser import CodeParser
    from .utils import util
    from .utils import util_ast
    from .utils import util_path


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


"""build your defined function for attributes"""


class AttrFns:
    @staticmethod
    def raw_ast_fn(filename, dest_filename, lang, start=0, end=-1):
        """code => raw_ast"""
        so_filename = os.path.join(os.path.expanduser(LIBS_DIR), '{}.so'.format(lang))
        parser = CodeParser(so_filename, lang)
        with open(filename, "r", encoding="UTF-8") as reader, open(dest_filename, 'w') as writer:
            reader.seek(start)
            line = safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                code = ujson.loads(line)
                raw_ast = parser.parse_raw_ast(code)
                # print(ujson.dumps(raw_ast, ensure_ascii=False), file=writer)
                line = safe_readline(reader)

    @staticmethod
    def ast_fn(filename, dest_filename, lang, start=0, end=-1):
        with open(filename, "r", encoding="UTF-8") as reader, open(dest_filename, 'w') as writer:
            reader.seek(start)
            line = safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                raw_ast = ujson.loads(line)
                ast = util_ast.convert(raw_ast)
                print(ujson.dumps(ast, ensure_ascii=False), file=writer)
                line = safe_readline(reader)

    @staticmethod
    def path_fn(filename, dest_filename, lang, start=0, end=-1):
        with open(filename, "r", encoding="UTF-8") as reader, open(dest_filename, 'w') as writer:
            reader.seek(start)
            line = safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                ast = ujson.loads(line)
                paths = util_path.ast_to_path(ast, MAX_PATH=300)
                print(ujson.dumps(paths, ensure_ascii=False), file=writer)
                line = safe_readline(reader)

    @staticmethod
    def sbt_fn(filename, dest_filename, lang, start=0, end=-1):
        with open(filename, "r", encoding="UTF-8") as reader, open(dest_filename, 'w') as writer:
            reader.seek(start)
            line = safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                ast = ujson.loads(line)
                padded_ast = util_ast.pad_leaf_node(ast)
                sbt = util_ast.parse_deepcom(padded_ast, util_ast.build_sbt_tree, to_lower=False)
                print(ujson.dumps(sbt, ensure_ascii=False), file=writer)
                line = safe_readline(reader)


def process(src_filename, tgt_filename, lang, num_workers=cpu_count()):
    def _cat(src_filenames, tgt_filename):
        cmd = 'cat {} > {}'.format(' '.join(src_filenames), tgt_filename)
        LOGGER.info(cmd)
        # run cat
        os.system(cmd)

    _src_filename = os.path.expanduser(src_filename)
    _tgt_filename = os.path.expanduser(tgt_filename)
    attr_fn = getattr(AttrFns, '{}_fn'.format(tgt_filename.split('.')[-1]))
    offsets = find_offsets(_src_filename, num_workers)
    with Pool(num_workers) as mpool:
        result = [
            mpool.apply_async(
                attr_fn,
                (_src_filename, _tgt_filename + str(idx), lang, offsets[idx], offsets[idx + 1])
            )
            for idx in range(num_workers)
        ]
        result = [res.get() for res in result]
    _cat(src_filenames=[_tgt_filename + str(idx) for idx in range(num_workers)], tgt_filename=tgt_filename)
    for idx in range(num_workers):
        os.remove(_tgt_filename + str(idx))


if __name__ == '__main__':
    """
    This script is to generate new attributes of code snippet.
    """
    parser = argparse.ArgumentParser(description="Download CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--language", "-l", default=LANGUAGES, type=list, help="languages constain [{}]".format(LANGUAGES),
    )
    parser.add_argument(
        "--flatten_dir", "-f", default=FLATTEN_DIR, type=str, help="data directory of flatten attribute",
    )
    parser.add_argument(
        "--attrs", "-a",
        default=['path'], type=list,
        help="attrs: raw_ast, ...",
    )
    parser.add_argument(
        "--cores", "-c", default=cpu_count(), type=int, help="cpu cores for flatten raw data attributes",
    )
    args = parser.parse_args()
    # print(args)

    raw_attrs, dest_attrs = [], []
    for attr in args.attrs:
        if attr == 'raw_ast':
            raw_attrs.append('code')
            dest_attrs.append(attr)
        elif attr == 'ast':
            raw_attrs.append('raw_ast')
            dest_attrs.append(attr)
        elif attr == 'path':
            raw_attrs.append('ast')
            dest_attrs.append(attr)
        # elif attr == 'sbt':
        #     raw_attrs.append('ast')
        #     dest_attrs.append(attr)

    args.language = ['ruby']
    for lang, mode in itertools.product(args.language, ['train', 'valid', 'test']):
        for src_attr, tgt_attr in zip(raw_attrs, dest_attrs):
            src_filename = os.path.join(args.flatten_dir, lang, '{}.{}'.format(mode, src_attr))
            tgt_filename = os.path.join(args.flatten_dir, lang, '{}.{}'.format(mode, tgt_attr))
            LOGGER.info('Generating {}'.format(tgt_filename))
            process(src_filename, tgt_filename, lang, num_workers=10)
