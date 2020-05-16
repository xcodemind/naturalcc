# -*- coding: utf-8 -*-
'''
ref: https://github.com/tech-srl/code2seq/blob/master/Python150kExtractor/extract.py
'''

from . import util
from . import constants
from typing import Dict
import re
import itertools
from ncc.data.constants import (
    CLS, S_SEP, H_SEP, T_SEP
)
from random import shuffle

MAX_PATH_LENTH = 8
MAX_PATH_WIDTH = 2


def __terminals(ast: Dict, node_index: str, ):
    stack, paths = [], []

    def dfs(v):
        stack.append(v)
        v_node = ast[v]

        child_nodes = util.get_tree_children_func(v_node)

        if len(child_nodes) == 0:
            # add leaf node's value
            paths.append((stack.copy(), v_node['children'][0]))
        else:
            # converse non-leaf node
            # add root node
            if v == constants.ROOT_NODE_NAME:
                paths.append((stack.copy(), v_node['node']))
            for child in v_node['children']:
                dfs(child)

        stack.pop()

    dfs(node_index)

    return paths


def __merge_terminals2_paths(v_path, u_path):
    s, n, m = 0, len(v_path), len(u_path)
    while s < min(n, m) and v_path[s] == u_path[s]:
        s += 1

    prefix = list(reversed(v_path[s:]))
    lca = v_path[s - 1]
    suffix = u_path[s:]

    return prefix, lca, suffix


def __raw_tree_paths(ast, node_index=constants.ROOT_NODE_NAME, ):
    tnodes = __terminals(ast, node_index, )

    tree_paths = []
    for (v_path, v_value), (u_path, u_value) in itertools.combinations(
            iterable=tnodes,
            r=2,
    ):
        prefix, lca, suffix = __merge_terminals2_paths(v_path, u_path)
        if (len(prefix) + 1 + len(suffix) <= MAX_PATH_LENTH) \
                and (abs(len(prefix) - len(suffix)) <= MAX_PATH_WIDTH):
            path = prefix + [lca] + suffix
            tree_path = v_value, path, u_value
            tree_paths.append(tree_path)

    return tree_paths


def __collect_sample(ast: Dict, MAX_PATH: int, to_lower: bool, split: bool):
    tree_paths = __raw_tree_paths(ast)
    contexts = []
    for tree_path in tree_paths:
        start, connector, finish = tree_path

        if split:
            start = [str.lower(token) if to_lower else token for token in util.split_identifier(start)]
            start = ' '.join(start)
            finish = [str.lower(token) if to_lower else token for token in util.split_identifier(finish)]
            finish = ' '.join(finish)

        connector = ' '.join([ast[v]['node'] for v in connector])
        # contexts.append([start, connector, finish])  # append a path
        contexts.append(' '.join([start, H_SEP, connector, T_SEP, finish]))
    try:
        assert len(contexts) > 0, Exception('ast\'s path is None')
        if len(contexts) > MAX_PATH:
            shuffle(contexts)
            contexts = contexts[:MAX_PATH]
        contexts = CLS + ' ' + ' {} '.format(S_SEP).join(contexts)
        return contexts
    except Exception as err:
        print(err)
        print(ast)
        return None


def ast_to_path(ast_tree: Dict, MAX_PATH: int, to_lower: bool, split: bool = True):
    return __collect_sample(ast_tree, MAX_PATH, to_lower, split)
