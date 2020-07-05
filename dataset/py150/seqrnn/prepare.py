# -*- coding: utf-8 -*-

import argparse
import os
import json
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


if __name__ == '__main__':
    """
    code string => code tokens and corresponding types sequence
    # for train data
    python -m dataset.py150.seqrnn.prepare -i ~/.ncc/py150/seqrnn/raw/python100k_train.txt -o ~/.ncc/py150/seqrnn/raw/train
    # for eval data  
    python -m dataset.py150.seqrnn.prepare -i ~/.ncc/py150/seqrnn/raw/python50k_eval.txt  -o ~/.ncc/py150/seqrnn/raw/test 
    """
    parser = argparse.ArgumentParser(description="Generate tokens and types from source code")
    parser.add_argument(
        "--in_file", "-i", default='~/.ncc/py150/seqrnn/raw/python100k_train.txt', help="source code file"
    )
    parser.add_argument(
        "--out_file", "-o", default='~/.ncc/py150/seqrnn/raw/train', help="source code file"
    )
    parser.add_argument(
        "--n_ctx", "-c", type=int, default=1000, help="Number of contexts for each dp"
    )
    args = parser.parse_args()

    in_file = args.in_file
    out_file_tok = args.out_file + '.tok'
    out_file_ids = args.out_file + '.ids'
    in_file, out_file_tok, out_file_ids = map(os.path.expanduser, (in_file, out_file_tok, out_file_ids,))
    raw_data_dir = os.path.dirname(in_file)
    n_ctx = args.n_ctx
    # MAX_SCRIPT_NUM = 100  # debug
    MAX_SCRIPT_NUM = 50000  # avoid out of memory


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
                token, ids = my_tokenize(code_str=code_string, n_ctx=n_ctx)
                return token, ids
        except:
            """ast parse expectation"""
            return


    with open(in_file, 'r', encoding='utf-8') as reader, \
            open(out_file_tok, 'w', encoding='utf-8') as tok_writer, \
            open(out_file_ids, 'w', encoding='utf-8') as ids_writer, \
            PPool() as thread_pool:
        file_stack = []
        for line in reader:
            raw_data_file = os.path.join(raw_data_dir, line.strip())
            file_stack.append(raw_data_file)
            if len(file_stack) >= MAX_SCRIPT_NUM:
                result = thread_pool.feed(parse_tokens_types, file_stack, one_params=True)
                result = filterfalse(lambda args: args is None, result)
                for token, ids in result:
                    if len(token) == len(ids) and len(ids) > 1 and \
                            (token is not None) and (ids is not None):
                        print(json.dumps(token), file=tok_writer)
                        print(json.dumps(ids), file=ids_writer)
                del file_stack
                file_stack = []

        if len(file_stack) > 0:
            result = thread_pool.feed(parse_tokens_types, file_stack, one_params=True)
            result = filterfalse(lambda args: args is None, result)
            for token, ids in result:
                if len(token) == len(ids) and len(ids) > 1 and \
                        (token is not None) and (ids is not None):
                    print(json.dumps(token), file=tok_writer)
                    print(json.dumps(ids), file=ids_writer)
            del file_stack
