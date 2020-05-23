# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Union, Any

import re
import ujson
import itertools
from ncc.data.constants import (
    H_SEP, T_SEP,
    PAD, CLS, S_SEP,
)

SPACE_NORMALIZER = re.compile(r"\s+")


# For compatibility
def tokenize_line(line: str) -> List[str]:
    """split string by regrex [\s+]"""
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def tokenize_string(line: str) -> List[str]:
    """split string by regrex [\s+]"""
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def tokenize_list(line: str) -> List[str]:
    """directly return tokenized data"""
    line = ujson.loads(line)
    return line


def tokenize_tree(line: str) -> List[str]:
    """get tree nodes' sub-tokens, and filter pad tokens because DGL requires sub-tokens' lengths should be same"""
    line = ujson.loads(line)
    leaf_node_tokens = []
    for node_info in line.values():
        if type(node_info['children'][-1]) == list:
            for token in node_info['children'][-1]:
                if token != PAD:
                    leaf_node_tokens.append(token)
                else:
                    break
    return leaf_node_tokens


def tokenize_border(line: str) -> List[str]:
    """process head/tail of path line"""
    line = ujson.loads(line)
    head, tail = line
    line = itertools.chain(*(head + tail))
    return line


def tokenize_body(line: str) -> List[str]:
    """process body of path line"""
    line = ujson.loads(line)
    line = itertools.chain(*line)
    return line


def tokenize_path(line: str, add_cls: bool = False) -> List[str]:
    """
    load path's head/body/tail tokens
    add_cls: add cls at 1st position of path
    """
    line = ujson.loads(line)
    paths = [CLS] if add_cls else []
    for idx, path in enumerate(line):
        head, body, tail = path
        path = head + [H_SEP] + body + [T_SEP] + tail
        paths.extend(path)
        if idx < len(line) - 1:
            paths.append(P_SEP)
    return paths


def CSN_tokenizer(modal: str):
    """
    CodeSearchNet modalities = [
        'bin_ast', 'code_tokens', 'docstring', 'func_name', 'method', 'path', 'sbt', 'tok',
        'code', 'comment', 'docstring_tokens', 'index', 'original_string', 'raw_ast', 'sbtao',
    ]

    There remain 3 types of data:
        1) data has been tokenized,
            e.g. 'docstring',  'code', ('index', 'original_string', 'method')
            we only recommend 'docstring'/'code' because
                'method' has already been tokinzed into 'func_name'
                'index' is a int number
                'original_string' includes code, docstring and noise information
        2) data remain as string type,
            e.g. 'code_tokens', 'func_name', 'sbt', 'tok', 'comment', 'docstring_tokens', 'sbtao'
        3) data has been tokenized but further serialized,
            e.g. path ([head], [center], [tail]), 'bin_ast', 'raw_ast' (dict)
    """
    # if modal in ['docstring', 'code', 'original_string', ]:
    #     return tokenize_string
    # elif modal in ['code_tokens', 'func_name', 'sbt', 'tok', 'comment', 'docstring_tokens', 'sbtao', ]:
    #     return tokenize_list
    # elif modal in ['bin_ast', 'raw_ast', ]:
    #     return tokenize_tree
    # elif modal == 'path':
    #     return tokenize_path
    # elif modal == 'body':
    #     return tokenize_body
    # elif modal == 'border':
    #     return tokenize_list
    # else:
    #     raise NotImplementedError
    return tokenize_string
