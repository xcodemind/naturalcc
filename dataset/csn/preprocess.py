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
                print(ujson.dumps(raw_ast, ensure_ascii=False), file=writer)
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
        # dest_filename_head, dest_filename_body, dest_filename_tail = \
        #     dest_filename + '.head', dest_filename + '.body', dest_filename + '.tail'
        # with open(filename, "r", encoding="UTF-8") as reader, open(dest_filename_head, 'w') as writer_head, \
        #     open(dest_filename_body, 'w') as writer_body, open(dest_filename_tail, 'w') as writer_tail:
        #     reader.seek(start)
        #     line = safe_readline(reader)
        #     while line:
        #         if end > 0 and reader.tell() > end:
        #             break
        #         ast = ujson.loads(line)
        #         paths = util_path.ast_to_path(ast, MAX_PATH=PATH_NUM)
        #         # copy paths size to PATH_NUM
        #         if len(paths) < PATH_NUM:
        #             supply_ids = list(range(len(paths))) * ((PATH_NUM - len(paths)) // len(paths)) \
        #                          + random.sample(range(len(paths)), ((PATH_NUM - len(paths)) % len(paths)))
        #             paths.extend([paths[idx] for idx in supply_ids])
        #             random.shuffle(paths)
        #         assert len(paths) == PATH_NUM
        #         for head, body, tail in paths:
        #             print(ujson.dumps(head, ensure_ascii=False), file=writer_head)
        #             print(ujson.dumps(body, ensure_ascii=False), file=writer_body)
        #             print(ujson.dumps(tail, ensure_ascii=False), file=writer_tail)
        #         line = safe_readline(reader)

        with open(filename, "r", encoding="UTF-8") as reader, open(dest_filename, 'w') as writer_head:
            reader.seek(start)
            line = safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                ast = ujson.loads(line)
                paths = util_path.ast_to_path(ast, MAX_PATH=PATH_NUM)
                # copy paths size to PATH_NUM
                if len(paths) < PATH_NUM:
                    supply_ids = list(range(len(paths))) * ((PATH_NUM - len(paths)) // len(paths)) \
                                 + random.sample(range(len(paths)), ((PATH_NUM - len(paths)) % len(paths)))
                    paths.extend([paths[idx] for idx in supply_ids])
                random.shuffle(paths)
                assert len(paths) == PATH_NUM
                for head_body_tail in itertools.chain(*paths):
                    print(ujson.dumps(head_body_tail, ensure_ascii=False), file=writer_head)
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
                ast = util_ast.value2children(ast)
                padded_ast = util_ast.pad_leaf_node(ast, MAX_SUB_TOKEN_LEN)
                root_idx = util_ast.get_root_idx(padded_ast)
                sbt = util_ast.build_sbt_tree(padded_ast, idx=root_idx)
                print(ujson.dumps(sbt, ensure_ascii=False), file=writer)
                line = safe_readline(reader)

    @staticmethod
    def sbtao_fn(filename, dest_filename, lang, start=0, end=-1):
        with open(filename, "r", encoding="UTF-8") as reader, open(dest_filename, 'w') as writer:
            reader.seek(start)
            line = safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                ast = ujson.loads(line)
                ast = util_ast.value2children(ast)
                padded_ast = util_ast.pad_leaf_node(ast, MAX_SUB_TOKEN_LEN)
                root_idx = util_ast.get_root_idx(padded_ast)
                sbt = util_ast.build_sbtao_tree(padded_ast, idx=root_idx)
                print(ujson.dumps(sbt, ensure_ascii=False), file=writer)
                line = safe_readline(reader)

    @staticmethod
    def bin_ast_fn(filename, dest_filename, lang, start=0, end=-1):
        with open(filename, "r", encoding="UTF-8") as reader, open(dest_filename, 'w') as writer:
            reader.seek(start)
            line = safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                ast = ujson.loads(line)
                ast = util_ast.value2children(ast)
                ast = util_ast.remove_root_with_uni_child(ast)
                root_idx = util_ast.get_root_idx(ast)
                ast = util_ast.delete_node_with_uni_child(ast, idx=root_idx)
                root_idx = util_ast.get_root_idx(ast)
                bin_ast = util_ast.binarize_tree(ast, idx=root_idx)  # to binary ast tree
                root_idx = util_ast.get_root_idx(ast)
                bin_ast = util_ast.reset_indices(bin_ast, root_idx)  # reset node indices
                bin_ast = util_ast.pad_leaf_node(bin_ast, MAX_SUB_TOKEN_LEN)
                print(ujson.dumps(bin_ast, ensure_ascii=False), file=writer)
                line = safe_readline(reader)


def process(src_filename, tgt_filename, lang, num_workers=cpu_count()):
    def _cat(src_filenames, tgt_filename):
        cmd = 'cat {} > {}'.format(' '.join(src_filenames), tgt_filename)
        # LOGGER.info(cmd)
        # run cat
        os.system(cmd)

    _src_filename = os.path.expanduser(src_filename)
    _tgt_filename = os.path.expanduser(tgt_filename)
    modality = tgt_filename.split('.')[-1]
    attr_fn = getattr(AttrFns, '{}_fn'.format(modality))
    offsets = find_offsets(_src_filename, num_workers)

    # # for debug
    # idx = 0
    # attr_fn(_src_filename, _tgt_filename + str(idx), lang, offsets[idx], offsets[idx + 1])

    with Pool(num_workers) as mpool:
        result = [
            mpool.apply_async(
                attr_fn,
                (_src_filename, _tgt_filename + str(idx), lang, offsets[idx], offsets[idx + 1])
            )
            for idx in range(num_workers)
        ]
        result = [res.get() for res in result]

    # if modality == 'path':
    #     _cat([_tgt_filename + str(idx) + '.head' for idx in range(num_workers)], tgt_filename + '.head')
    #     _cat([_tgt_filename + str(idx) + '.body' for idx in range(num_workers)], tgt_filename + '.body')
    #     _cat([_tgt_filename + str(idx) + '.tail' for idx in range(num_workers)], tgt_filename + '.tail')
    #     for idx in range(num_workers):
    #         os.remove(_tgt_filename + str(idx) + '.head')
    #         os.remove(_tgt_filename + str(idx) + '.body')
    #         os.remove(_tgt_filename + str(idx) + '.tail')
    # else:
    #     _cat([_tgt_filename + str(idx) for idx in range(num_workers)], tgt_filename)
    #     for idx in range(num_workers):
    #         os.remove(_tgt_filename + str(idx))

    _cat([_tgt_filename + str(idx) for idx in range(num_workers)], tgt_filename)
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
        # default=['raw_ast', 'ast', 'path', 'sbt', 'sbtao', 'bin_ast'], type=list,
        default=['path'], type=list,
        help="attrs: raw_ast, ...",
    )
    parser.add_argument(
        "--cores", "-c", default=cpu_count(), type=int, help="cpu cores for flatten raw data attributes",
    )
    args = parser.parse_args()
    # print(args)

    """
    a mapping to generate new attributes of code snippet.
    Examples:
        "raw_ast" <= "code",    # raw_ast, an AST contains all info of a code, e.g. comment, single root node, ...
        "ast" <= "raw_ast",     # ast, saving leaf nodes into "value" nodes and non-leaf nodes into "children" nodes
        "path" <= "ast",        # path, a path from a leaf node to another leaf node 
        "sbt" <= "raw_ast",     # sbt, a depth first traversal path of an AST, tokenize leaf node and padding with <PAD>(for DGL Lib.)
        "sbtao" <= "st'",       # sbtao, an improved depth first traversal path of an AST, tokenize leaf node and padding with <PAD>(for DGL Lib.)
        "bin_ast" <= "raw_ast", # bin_ast, an sophisticated binary AST, remove nodes with single child, tokenize leaf node and padding with <PAD>(for DGL Lib.)
        
    """

    dest_raw_attrs = {
        'raw_ast': 'code',
        'ast': 'raw_ast',
        'path': 'ast',
        'sbt': 'raw_ast',
        'sbtao': 'raw_ast',
        'bin_ast': 'raw_ast',
    }

    args.language = ['ruby']
    for lang, mode in itertools.product(args.language, MODES):
        for tgt_attr in args.attrs:
            src_attr = dest_raw_attrs[tgt_attr]
            src_filename = os.path.join(args.flatten_dir, lang, '{}.{}'.format(mode, src_attr))
            tgt_filename = os.path.join(args.flatten_dir, lang, '{}.{}'.format(mode, tgt_attr))
            LOGGER.info('Generating {}'.format(tgt_filename))
            process(src_filename, tgt_filename, lang, num_workers=args.cores)
