# -*- coding: utf-8 -*-
'''
ref: https://github.com/tech-srl/code2seq/blob/master/Python150kExtractor/extract.py
'''

from . import util
from . import constants
from typing import Dict
import re
import itertools

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


def __delim_name(name):
    def camel_case_split(identifier):
        matches = re.finditer(
            '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
            identifier,
        )
        return [m.group(0) for m in matches]

    blocks = []
    for underscore_block in name.split('_'):
        blocks.extend(camel_case_split(underscore_block))

    return '|'.join(block.lower() for block in blocks)


def __collect_sample(ast: Dict, MAX_PATH: int, ):
    tree_paths = __raw_tree_paths(ast, )
    contexts = []
    for tree_path in tree_paths:
        start, connector, finish = tree_path

        # start, finish = __delim_name(start), __delim_name(finish)
        new_start, new_finish = [str.lower(token) for token in util.split_identifier(start)], \
                                [str.lower(token) for token in util.split_identifier(finish)]

        connector = [ast[v]['node'] for v in connector]

        if len(start) > 0 and len(connector) > 0 and len(finish) > 0:
            contexts.append([new_start, connector, new_finish])
        else:
            # LOGGER.error(tree_path)
            pass
    try:
        assert len(contexts) > 0, Exception('ast\'s path is None')
        return contexts[:MAX_PATH]
    except Exception as err:
        print(err)
        print(ast)
        return None


def ast_to_path(ast_tree: Dict, MAX_PATH: int):
    return __collect_sample(ast_tree, MAX_PATH)
