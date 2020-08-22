# -*- coding: utf-8 -*-

import os
import re
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
        RAW_DATA_DIR, LIBS_DIR, REFINE_DIR, FLATTEN_DIR,
        LOGGER,
    )
    from dataset.csn.parser._parser import CodeParser
    from dataset.csn.utils import (util, util_ast, util_path, util_traversal)
except ImportError:
    from . import (
        LANGUAGES, MODES,
        RAW_DATA_DIR, LIBS_DIR, REFINE_DIR, FLATTEN_DIR,
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


"""build your defined function for attributes"""


class AttrFns:
    @staticmethod
    def code_tokens_fn(filename, dest_filename, idx, start=0, end=-1, *args):
        """code => raw_ast"""
        kwargs = args[0][0]  # canot feed dict parameters in multi-processing

        dest_filename = dest_filename + str(idx)
        os.makedirs(os.path.dirname(dest_filename), exist_ok=True)
        with open(filename, "r", encoding="UTF-8") as reader, open(dest_filename, 'w') as writer:
            reader.seek(start)
            line = safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                code_tokens = ujson.loads(line)
                if code_tokens:
                    # filter comment in code_tokens, eg. //***\n /* */\n
                    code_tokens = [token for token in code_tokens
                                   if not (str.startswith(token, '//') or str.startswith(token, '#') or \
                                           (str.startswith(token, '/*') and str.endswith(token, '*/')))
                                   ]

                    if not all(str.isascii(token) for token in code_tokens):
                        code_tokens = None
                    if code_tokens is None or len(code_tokens) < 1:
                        code_tokens = None
                else:
                    code_tokens = None

                print(ujson.dumps(code_tokens, ensure_ascii=False), file=writer)
                line = safe_readline(reader)

    @staticmethod
    def docstring_tokens_fn(filename, dest_filename, idx, start=0, end=-1, *args):
        """code => raw_ast"""
        kwargs = args[0][0]  # canot feed dict parameters in multi-processing

        dest_filename = dest_filename + str(idx)
        os.makedirs(os.path.dirname(dest_filename), exist_ok=True)
        with open(filename, "r", encoding="UTF-8") as reader, open(dest_filename, 'w') as writer:
            reader.seek(start)
            line = safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                docstring_tokens = ujson.loads(line)
                if docstring_tokens:
                    docstring_tokens = [
                        token for token in docstring_tokens \
                        if not (re.match(r'[\-|\*|\=|\~]{2,}', token) or re.match(r'<.*?>', token))
                    ]
                    if not all(str.isascii(token) for token in docstring_tokens):
                        docstring_tokens = None
                    if (docstring_tokens is None) or not (3 < len(docstring_tokens) <= 50):
                        docstring_tokens = None
                else:
                    docstring_tokens = None
                print(ujson.dumps(docstring_tokens, ensure_ascii=False), file=writer)
                line = safe_readline(reader)

    @staticmethod
    def raw_ast_fn(filename, dest_filename, idx, start=0, end=-1, *args):
        """code => raw_ast"""
        kwargs = args[0][0]  # canot feed dict parameters in multi-processing
        lang = kwargs.get('lang')
        so_dir = kwargs.get('so_dir')

        so_filename = os.path.join(os.path.expanduser(so_dir), '{}.so'.format(lang))
        parser = CodeParser(so_filename, lang)
        dest_filename = dest_filename + str(idx)
        with open(filename, "r", encoding="UTF-8") as reader, open(dest_filename, 'w') as writer:
            reader.seek(start)
            line = safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                code = ujson.loads(line)
                if code:
                    raw_ast = parser.parse_raw_ast(code)
                else:
                    raw_ast = None
                print(ujson.dumps(raw_ast, ensure_ascii=False), file=writer)
                line = safe_readline(reader)

    @staticmethod
    def ast_fn(filename, dest_filename, idx, start=0, end=-1, *args):
        kwargs = args[0][0]  # canot feed dict parameters in multi-processing

        dest_filename = dest_filename + str(idx)
        with open(filename, "r", encoding="UTF-8") as reader, open(dest_filename, 'w') as writer:
            reader.seek(start)
            line = safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                raw_ast = ujson.loads(line)
                if raw_ast:
                    ast = util_ast.convert(raw_ast)
                else:
                    ast = None
                print(ujson.dumps(ast, ensure_ascii=False), file=writer)
                line = safe_readline(reader)

    @staticmethod
    def path_fn(filename, dest_filename, idx, start=0, end=-1, *args):
        kwargs = args[0][0]  # canot feed dict parameters in multi-processing

        dest_filename_terminals, dest_filename = dest_filename + '.terminals' + str(idx), dest_filename + str(idx)
        with open(filename, "r", encoding="UTF-8") as reader, open(dest_filename_terminals, 'w') as writer_terminals, \
            open(dest_filename, 'w') as writer:
            reader.seek(start)
            line = safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                ast = ujson.loads(line)
                if ast:
                    paths = util_path.ast_to_path(ast, MAX_PATH=PATH_NUM)
                    if paths is None:
                        paths = [[None] * 3] * PATH_NUM
                    else:
                        # copy paths size to PATH_NUM
                        if len(paths) < PATH_NUM:
                            supply_ids = list(range(len(paths))) * ((PATH_NUM - len(paths)) // len(paths)) \
                                         + random.sample(range(len(paths)), ((PATH_NUM - len(paths)) % len(paths)))
                            paths.extend([paths[idx] for idx in supply_ids])
                    random.shuffle(paths)
                    assert len(paths) == PATH_NUM
                    head, body, tail = zip(*paths)
                else:
                    head, body, tail = [None] * PATH_NUM, [None] * PATH_NUM, [None] * PATH_NUM
                # terminals
                for terminal in itertools.chain(*zip(head, tail)):
                    print(ujson.dumps(terminal, ensure_ascii=False), file=writer_terminals)
                # path
                for b in body:
                    print(ujson.dumps(b, ensure_ascii=False), file=writer)
                line = safe_readline(reader)

    @staticmethod
    def sbt_fn(filename, dest_filename, idx, start=0, end=-1, *args):
        kwargs = args[0][0]  # canot feed dict parameters in multi-processing

        dest_filename = dest_filename + str(idx)
        with open(filename, "r", encoding="UTF-8") as reader, open(dest_filename, 'w') as writer:
            reader.seek(start)
            line = safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                ast = ujson.loads(line)
                if ast:
                    ast = util_ast.value2children(ast)
                    padded_ast = util_ast.pad_leaf_node(ast, MAX_SUB_TOKEN_LEN)
                    root_idx = util_ast.get_root_idx(padded_ast)
                    sbt = util_ast.build_sbt_tree(padded_ast, idx=root_idx)
                else:
                    sbt = None
                print(ujson.dumps(sbt, ensure_ascii=False), file=writer)
                line = safe_readline(reader)

    @staticmethod
    def sbtao_fn(filename, dest_filename, idx, start=0, end=-1, *args):
        kwargs = args[0][0]  # canot feed dict parameters in multi-processing

        dest_filename = dest_filename + str(idx)
        with open(filename, "r", encoding="UTF-8") as reader, open(dest_filename, 'w') as writer:
            reader.seek(start)
            line = safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                ast = ujson.loads(line)
                if ast:
                    ast = util_ast.value2children(ast)
                    padded_ast = util_ast.pad_leaf_node(ast, MAX_SUB_TOKEN_LEN)
                    root_idx = util_ast.get_root_idx(padded_ast)
                    sbt = util_ast.build_sbtao_tree(padded_ast, idx=root_idx)
                else:
                    sbt = None
                print(ujson.dumps(sbt, ensure_ascii=False), file=writer)
                line = safe_readline(reader)

    @staticmethod
    def binary_ast_fn(filename, dest_filename, idx, start=0, end=-1, *args):
        kwargs = args[0][0]  # canot feed dict parameters in multi-processing

        dest_filename = dest_filename + str(idx)
        with open(filename, "r", encoding="UTF-8") as reader, open(dest_filename, 'w') as writer:
            reader.seek(start)
            line = safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                ast = ujson.loads(line)
                if ast:
                    try:
                        ast = util_ast.value2children(ast)
                        ast = util_ast.remove_root_with_uni_child(ast)
                        root_idx = util_ast.get_root_idx(ast)
                        ast = util_ast.delete_node_with_uni_child(ast, idx=root_idx)
                        root_idx = util_ast.get_root_idx(ast)
                        bin_ast = util_ast.binarize_tree(ast, idx=root_idx)  # to binary ast tree
                        root_idx = util_ast.get_root_idx(ast)
                        bin_ast = util_ast.reset_indices(bin_ast, root_idx)  # reset node indices
                        bin_ast = util_ast.pad_leaf_node(bin_ast, MAX_SUB_TOKEN_LEN)
                    except RecursionError:
                        LOGGER.info('RecursionError, ignore this tree')
                        bin_ast = None
                    except Exception as err:
                        print(err)
                        bin_ast = None
                else:
                    bin_ast = None
                print(ujson.dumps(bin_ast, ensure_ascii=False), file=writer)
                line = safe_readline(reader)

    @staticmethod
    def traversal_fn(filename, dest_filename, idx, start=0, end=-1, *args):
        kwargs = args[0][0]  # canot feed dict parameters in multi-processing

        dest_filename = dest_filename + str(idx)
        with open(filename, "r", encoding="UTF-8") as reader, open(dest_filename, 'w') as writer:
            reader.seek(start)
            line = safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                ast = ujson.loads(line)
                if ast:
                    ast_traversal = util_traversal.get_dfs(ast)
                else:
                    ast_traversal = None
                print(ujson.dumps(ast_traversal, ensure_ascii=False), file=writer)
                line = safe_readline(reader)


def process(src_filename, tgt_filename, num_workers=cpu_count(), **kwargs):
    _src_filename = os.path.expanduser(src_filename)
    _tgt_filename = os.path.expanduser(tgt_filename)
    modality = tgt_filename.split('.')[-1]
    attr_fn = getattr(AttrFns, '{}_fn'.format(modality))
    offsets = find_offsets(_src_filename, num_workers)

    # # for debug
    # idx = 0
    # attr_fn(_src_filename, _tgt_filename, idx, offsets[idx], offsets[idx + 1], [kwargs])

    with Pool(num_workers) as mpool:
        result = [
            mpool.apply_async(
                attr_fn,
                (_src_filename, _tgt_filename, idx, offsets[idx], offsets[idx + 1], [kwargs])
            )
            for idx in range(num_workers)
        ]
        result = [res.get() for res in result]

    def _cat_and_remove(_tgt_filename, num_workers, tgt_filename):
        src_filenames = [_tgt_filename + str(idx) for idx in range(num_workers)]
        cmd = 'cat {} > {}'.format(' '.join(src_filenames), tgt_filename)
        LOGGER.info(cmd)
        os.system(cmd)
        for _src_fl in src_filenames:
            os.remove(_src_fl)

    _cat_and_remove(_tgt_filename, num_workers, tgt_filename)
    if modality == 'path':
        _cat_and_remove(_tgt_filename + '.terminals', num_workers, tgt_filename + '.terminals')


if __name__ == '__main__':
    """
    This script is to generate new attributes of code snippet.
    """
    parser = argparse.ArgumentParser(description="Download CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--language", "-l", default=LANGUAGES, type=str, nargs='+', help="languages constain [{}]".format(LANGUAGES),
    )
    parser.add_argument(
        "--flatten_dir", "-f", default=FLATTEN_DIR, type=str, help="data directory of flatten attribute(load)",
    )
    parser.add_argument(
        "--refine_dir", "-r", default=REFINE_DIR, type=str, help="data directory of refine attribute(save)",
    )
    parser.add_argument(
        "--so_dir", "-s", default=LIBS_DIR, type=str, help="library directory of so file",
    )
    parser.add_argument(
        "--attrs", "-a",
        default=[
            'code_tokens', 'docstring_tokens',
            'raw_ast', 'ast', 'path', 'sbt', 'sbtao', 'binary_ast', 'traversal',
        ],
        type=str, nargs='+', help="attrs: raw_ast, ...",
    )
    parser.add_argument(
        "--cores", "-c", default=cpu_count(), type=int, help="cpu cores for flatten raw data attributes",
    )
    args = parser.parse_args()
    print(args)

    """
    a mapping to generate new attributes of code snippet.
    Examples:
        "raw_ast" <= "code",    # raw_ast, an AST contains all info of a code, e.g. comment, single root node, ...
        "ast" <= "raw_ast",     # ast, saving leaf nodes into "value" nodes and non-leaf nodes into "children" nodes
        "path" <= "ast",        # path, a path from a leaf node to another leaf node 
        "sbt" <= "raw_ast",     # sbt, a depth first traversal path of an AST, tokenize leaf node and padding with <PAD>(for DGL Lib.)
        "sbtao" <= "sbt'",       # sbtao, an improved depth first traversal path of an AST, tokenize leaf node and padding with <PAD>(for DGL Lib.)
        "binary_ast" <= "raw_ast", # bin_ast, an sophisticated binary AST, remove nodes with single child, tokenize leaf node and padding with <PAD>(for DGL Lib.)
        "traversal" <= "ast",   #
    """

    dest_raw_attrs = {
        'code_tokens': 'code_tokens',
        'docstring_tokens': 'docstring_tokens',
        'raw_ast': 'code',
        'ast': 'raw_ast',
        'path': 'ast',
        'sbt': 'raw_ast',
        'sbtao': 'raw_ast',
        'binary_ast': 'raw_ast',
        'traversal': 'ast',
    }

    for lang, mode in itertools.product(args.language, MODES):
        for tgt_attr in args.attrs:
            src_attr = dest_raw_attrs[tgt_attr]
            src_filename = os.path.join(args.flatten_dir, lang, '{}.{}'.format(mode, src_attr))
            tgt_filename = os.path.join(args.refine_dir, lang, '{}.{}'.format(mode, tgt_attr))
            LOGGER.info('Generating {}'.format(tgt_filename))
            process(src_filename, tgt_filename, num_workers=args.cores,
                    lang=lang, so_dir=args.so_dir)
