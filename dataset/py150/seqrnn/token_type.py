# -*- coding: utf-8 -*-

import argparse
import os
import ujson
import ast
from itertools import filterfalse

from dataset.py150.seqrnn.astunparser import Unparser
from ncc.utils.mp_ppool import PPool

from collections import namedtuple

SrcASTToken = namedtuple("SrcASTToken", "text type")


def get_leaf_ids(types_):
    ids = {"leaf_ids": []}
    for i, v in enumerate(types_):
        if v is not None:
            ids["leaf_ids"].append(i)
    return ids


def get_value_ids(types_):
    ids = {"attr_ids": [], "num_ids": [], "name_ids": [], "param_ids": []}
    for i, v in enumerate(types_):
        if v == "attr":
            ids["attr_ids"].append(i)
        elif v == "Num":
            ids["num_ids"].append(i)
        elif v in {"NameStore", "NameLoad"}:
            ids["name_ids"].append(i)
        elif v == "NameParam":
            ids["param_ids"].append(i)
    return ids


class MyListFile(list):
    def write(self, text, type=None):
        text = text.strip()
        if len(text) > 0:
            self.append(SrcASTToken(text, type))

    def flush(self):
        pass

    def transpose(self, max_len):
        tokens = [tt.text for tt in self]
        types_ = [tt.type for tt in self]
        return tokens, types_


def my_tokenize(code_str, n_ctx):
    t = ast.parse(code_str)
    lst = MyListFile()
    Unparser(t, lst)
    return lst.transpose(n_ctx)


def parse_tokens_types(filename):
    """
    Examples:
        def add(a, b):\n  return a + b
    Returns
        ['def', 'add', '(', 'a', ',', 'b', ')', ':', 'return', '(', 'a', '+', 'b', ')']
        [None, 'FunctionDef', None, 'arg', None, 'arg', None, None, None, None, 'NameLoad', None, 'NameLoad', None]
    """
    try:
        with open(filename, 'r', encoding='utf-8') as reader:
            code_string = reader.read().strip()
            # code_string = 'def add(a, b):\n  return a + b'
            tokens, types = my_tokenize(code_str=code_string, n_ctx=n_ctx)
            return tokens, types
    except:
        """ast parse expectation"""
        return


if __name__ == '__main__':
    """
    code string => code tokens and corresponding types sequence
    """
    parser = argparse.ArgumentParser(description="Generate tokens and types from source code")
    parser.add_argument(
        "--in_file", "-i", default='~/.ncc/py150/seqrnn/raw/train.txt', help="source code file"
    )
    parser.add_argument(
        "--n_ctx", "-c", type=int, default=1000, help="Number of contexts for each dp"
    )
    args = parser.parse_args()

    in_file = args.in_file
    out_file_tok = in_file[:in_file.find('.txt')] + '.tok'
    out_file_type = in_file[:in_file.find('.txt')] + '.type'
    in_file, out_file_tok, out_file_type = map(os.path.expanduser, (in_file, out_file_tok, out_file_type,))
    raw_data_dir = os.path.dirname(in_file)
    n_ctx = args.n_ctx
    MAX_SCRIPT_NUM = 50000  # avoid out of memory
    # MAX_SCRIPT_NUM = 100  # debug

    with open(in_file, 'r', encoding='utf-8') as reader, \
            open(out_file_tok, 'w', encoding='utf-8') as tok_writer, \
            open(out_file_type, 'w', encoding='utf-8') as type_writer, \
            PPool() as thread_pool:
        file_stack = []
        for line in reader:
            raw_data_file = os.path.join(raw_data_dir, line.strip())
            file_stack.append(raw_data_file)
            if len(file_stack) >= MAX_SCRIPT_NUM:
                result = thread_pool.feed(parse_tokens_types, file_stack, one_params=True)
                result = filterfalse(lambda args: args is None, result)
                for tokens, types in result:
                    if len(tokens) == len(types) and len(tokens) > 1 and \
                            (tokens is not None) and (types is not None):
                        tok_writer.write(ujson.dumps(tokens) + '\n')
                        type_writer.write(ujson.dumps(types) + '\n')
                del file_stack
                file_stack = []

        if len(file_stack) > 0:
            result = thread_pool.feed(parse_tokens_types, file_stack, one_params=True)
            result = filterfalse(lambda args: args is None, result)
            for tokens, types in result:
                if len(tokens) == len(types) and len(tokens) > 1 and \
                        (tokens is not None) and (types is not None):
                    tok_writer.write(ujson.dumps(tokens) + '\n')
                    type_writer.write(ujson.dumps(types) + '\n')
            del file_stack
