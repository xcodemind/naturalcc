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
    PAD, CLS, P_SEP,
)

from dpu_utils.codeutils import split_identifier_into_parts

SPACE_NORMALIZER = re.compile(r"\s+")
IDENTIFIER_TOKEN_REGEX = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')


def tokenize_string(line: str) -> List[str]:
    tokens = SPACE_NORMALIZER.sub(" ", line).strip().split()
    tokens = [split_identifier_into_parts(tok) if IDENTIFIER_TOKEN_REGEX.match(tok) else [tok] for tok in tokens]
    tokens = list(itertools.chain(*tokens))
    return tokens


# For compatibility
def tokenize_line(line: str) -> List[str]:
    """split string by regrex [\s+]"""
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


# def tokenize_string(line: str) -> List[str]:
#     """split string by regrex [\s+]"""
#     line = SPACE_NORMALIZER.sub(" ", line)
#     line = line.strip()
#     return line.split()


def tokenize_list(line: str) -> List[str]:
    """directly return tokenized data"""
    line = ujson.loads(line)
    return line


def tokenize_tree(line: str) -> List[str]:
    """get tree nodes' sub-tokens, and filter pad tokens because DGL requires sub-tokens' lengths should be same"""
    line = ujson.loads(line)
    leaf_node_tokens = []
    for node_info in line.values():
        if isinstance(node_info['children'][-1], list):
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


def tokenize_path(line: str) -> List[str]:
    """
    load path's head/body/tail tokens
    add_cls: add cls at 1st position of path
    """
    line = ujson.loads(line)
    tokens = list(itertools.chain(*line))
    return tokens


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


# ======================== py150 dataset ======================== #

def tokenize_type_value(line: str) -> List[str]:
    """return non-leaf node type and leaf node value of py150 dfs ast"""
    line = ujson.loads(line)
    type_and_values = []
    for node in line:
        if 'type' in node:
            type_and_values.append(node['type'])
        elif 'value' in node:
            type_and_values.append(node['value'])
        else:
            pass
    return type_and_values
