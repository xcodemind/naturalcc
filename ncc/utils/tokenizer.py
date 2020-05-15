# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Union, Any

import re
import ujson
import itertools
from ncc.data.constants import PAD

SPACE_NORMALIZER = re.compile(r"\s+")


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


# def tokenize_path(line: str):
#     """load path's border and center tokens"""
#     line = ujson.loads(line)
#     head_list, center_list, tail_list = zip(*line)
#     border_list = head_list + tail_list
#     border_list, center_list, = map(
#         lambda lst: itertools.chain(*lst),
#         (border_list, center_list,)
#     )
#     return border_list, center_list


def CSN_tokinzer(modal: str):
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
    if modal in ['docstring', 'code', 'original_string', ]:
        return tokenize_string
    elif modal in ['code_tokens', 'func_name', 'sbt', 'tok', 'comment', 'docstring_tokens', 'sbtao', ]:
        return tokenize_list
    elif modal in ['bin_ast', 'raw_ast', ]:
        return tokenize_tree
    elif modal == 'body':
        return tokenize_body
    elif modal == 'border':
        return tokenize_list
    else:
        raise NotImplementedError


# def tokinzer_returns(func: Any) -> int:
#     """return the number of a tokenizer function's return values"""
#     if func in [tokenize_path, ]:
#         return 2
#     elif func in [tokenize_string, tokenize_list, tokenize_multi_list, ]:
#         return 1
#     else:
#         raise NotImplementedError('No such function in {}'.format(__file__))


__all__ = (
    CSN_tokinzer,
    # tokinzer_returns,
)
