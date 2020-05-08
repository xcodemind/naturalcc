# -*- coding: utf-8 -*-
from typing import List, Dict, Any

import sys
from copy import deepcopy

from dataset.codesearchnet.utils import constants
from dataset.codesearchnet.utils import util

# ignore those ast whose size is too large. Therefore set it as a small number
sys.setrecursionlimit(constants.RECURSION_DEPTH)  # recursion depth


def delete_node_with_single_node(ast_tree: Dict) -> Dict:
    '''delete nodes who has only one child'''

    def dfs(node_ind):
        cur_node = ast_tree[node_ind]
        child_node_indices = util.get_tree_children_func(cur_node)

        # each ast tree generally is parsed from a method, so it has a "program" root node and a "method" node
        # therefore, if current node is the root node with single child, we do not delete it
        while len(child_node_indices) == 1 and cur_node['parent'] is not None:
            # update its parent's children
            parent_node = ast_tree[cur_node['parent']]
            del_ind = parent_node['children'].index(node_ind)
            del parent_node['children'][del_ind]
            child_ind = child_node_indices[0]
            # update its children's parent to its parent
            ast_tree[child_ind]['parent'] = cur_node['parent']
            # update its parent's children
            parent_node['children'].insert(del_ind, child_ind)
            # elete itself
            ast_tree.pop(node_ind)

            # update current info
            node_ind = child_ind
            cur_node = ast_tree[node_ind]
            child_node_indices = util.get_tree_children_func(cur_node)

        if len(child_node_indices) == 0:
            return

        for child_name in child_node_indices:
            dfs(child_name)

    dfs(constants.ROOT_NODE_NAME)
    return ast_tree


def binarize_tree(ast_tree: Dict) -> Dict:
    '''ast tree -> binary ast tree'''
    last_node_ind = util.last_index(ast_tree)

    def dfs(cur_node_ind):
        cur_node = ast_tree[cur_node_ind]
        child_node_indices = util.get_tree_children_func(cur_node)

        if len(child_node_indices) > 2:
            # add new node
            nonlocal last_node_ind
            last_node_ind += 1
            new_node_ind = constants.NODE_FIX + str(last_node_ind)
            new_node = {
                'node': constants.NODE_TMP,
                'parent': cur_node_ind,
                'children': child_node_indices[1:],
            }
            ast_tree[new_node_ind] = new_node
            # update node's children info
            cur_node['children'] = [child_node_indices[0], new_node_ind]
            # update other childen nodes' parent info
            for child_name in child_node_indices[1:]:
                if child_name.startswith(constants.NODE_FIX) and child_name in ast_tree:
                    ast_tree[child_name]['parent'] = new_node_ind
            # update current node's children info
            child_node_indices = util.get_tree_children_func(cur_node)

        if len(child_node_indices) == 0:
            return

        for child_name in cur_node['children']:
            dfs(child_name)

    dfs(constants.ROOT_NODE_NAME)
    return ast_tree


def split_and_pad_token(token: str, MAX_TOKEN_LIST_LEN: int, to_lower: bool = True,
                        PAD_TOKEN: str = constants.PAD_WORD) -> List:
    '''
    split token and pad it with PAD_TOKEN till reach MAX_TOKEN_LIST_LEN
    e.g. VariableName ->  [VariableName, [Variable, Name, PAD_TOKEN, PAD_TOKEN, ...]]
    :param token: raw token
    :param MAX_TOKEN_LIST_LEN: max pad length
    :param to_lower:
    :return:
    '''
    tokens = util.split_identifier(token)
    if to_lower:
        tokens = util.lower(tokens)
    tokens.extend([PAD_TOKEN for _ in range(MAX_TOKEN_LIST_LEN - len(tokens))])
    return tokens


def pad_leaf_node(ast_tree: Dict, MAX_LEN: int, to_lower: bool = True, PAD_TOKEN: str = constants.PAD_WORD) -> Dict:
    '''
    pad leaf node's child into [XX, [XX, ...]]
    :param ast_tree:
    :param MAX_LEN: max pad length
    :return:
    '''
    for key, node in ast_tree.items():
        if len(node['children']) == 1 and (not str.startswith(node['children'][0], constants.NODE_FIX)):
            ast_tree[key]['children'].append(
                split_and_pad_token(ast_tree[key]['children'][0], MAX_LEN, to_lower, PAD_TOKEN)
            )
    return ast_tree


def build_sbt_tree(ast_tree: Dict, node_ind: str, to_lower: bool) -> List:
    '''
    build structure-based traversal SBT tree
    ref: Deep Code Comment Generation
    '''
    if len(ast_tree[node_ind]['children']) > 1 and type(ast_tree[node_ind]['children'][1]) == list:
        token = ast_tree[node_ind]['node'] + '_' + ast_tree[node_ind]['children'][0]
        if to_lower:
            token = token.lower()
        seq = [constants.SBT_PARENTHESES[0], token, constants.SBT_PARENTHESES[1], token]
    else:
        token = ast_tree[node_ind]['node']
        if to_lower:
            token = token.lower()
        seq = [constants.SBT_PARENTHESES[0], token]
        for child_ind in ast_tree[node_ind]['children']:
            seq += build_sbt_tree(ast_tree, child_ind, to_lower)
        seq += [constants.SBT_PARENTHESES[1], token]
    return seq


def build_sbtao_tree(ast_tree: Dict, node_ind: str, to_lower: bool) -> List:
    '''
    build structure-based traversal SBT tree
    ref: Deep Code Comment Generation
    :return:
    '''
    if len(ast_tree[node_ind]['children']) > 1 and type(ast_tree[node_ind]['children'][1]) == list:
        token = ast_tree[node_ind]['node'] + '_' + '<other>'
        if to_lower:
            token = token.lower()
        seq = [constants.SBT_PARENTHESES[0], token, constants.SBT_PARENTHESES[1], token]
    else:
        token = ast_tree[node_ind]['node']
        if to_lower:
            token = token.lower()
        seq = [constants.SBT_PARENTHESES[0], token]
        for child_ind in ast_tree[node_ind]['children']:
            seq += build_sbtao_tree(ast_tree, child_ind, to_lower)
        seq += [constants.SBT_PARENTHESES[1], token]
    return seq


def parse_deepcom(ast_tree: dict, sbt_func: Any, to_lower: bool):
    sbt_seq = sbt_func(ast_tree, constants.ROOT_NODE_NAME, to_lower)
    return sbt_seq


def delete_single_child_ndoe(ast_tree: Dict) -> Dict:
    '''
    delete nodes with single child node
    :param ast_tree:
    :return:
    '''

    def dfs(node_ind):
        cur_node = ast_tree[node_ind]
        child_node_indices = util.get_tree_children_func(cur_node)

        # each ast tree generally is parsed from a method, so it has a "program" root node and a "method" node
        # therefore, if current node is the root node with single child, we do not delete it
        while len(child_node_indices) == 1 and cur_node['parent'] is not None:
            # update its parent's children
            parent_node = ast_tree[cur_node['parent']]
            del_ind = parent_node['children'].index(node_ind)
            del parent_node['children'][del_ind]
            child_ind = child_node_indices[0]
            # update its children's parent to its parent
            ast_tree[child_ind]['parent'] = cur_node['parent']
            # update its parent's children
            parent_node['children'].insert(del_ind, child_ind)
            # elete itself
            ast_tree.pop(node_ind)

            # update current info
            node_ind = child_ind
            cur_node = ast_tree[node_ind]
            child_node_indices = util.get_tree_children_func(cur_node)

        if len(child_node_indices) == 0:
            return

        for child_name in child_node_indices:
            dfs(child_name)

    dfs(constants.ROOT_NODE_NAME)
    return ast_tree


def reset_indices(ast_tree: Dict) -> Dict:
    '''rename ast tree's node indices with consecutive indices'''
    new_ind = 1
    root_ind = 1
    while 1:
        root_node_ind = constants.NODE_FIX + str(root_ind)
        if root_node_ind in ast_tree:
            break
        else:
            root_ind += 1

    def new_ndoe_name():
        nonlocal new_ind
        new_name = '_' + constants.NODE_FIX + str(new_ind)
        new_ind += 1
        return new_name

    def dfs(cur_node_ind):
        cur_node = ast_tree[cur_node_ind]
        # change from cur_node_ind to new_cur_node_ind
        # copy a same node with new name
        new_cur_node_ind = new_ndoe_name()
        ast_tree[new_cur_node_ind] = deepcopy(cur_node)

        # update its parent's child nodes
        if cur_node['parent'] is None:
            pass
        else:
            parent_node = ast_tree[cur_node['parent']]
            parent_node['children'][parent_node['children'].index(cur_node_ind)] = new_cur_node_ind

        if cur_node['children'][0].startswith(constants.NODE_FIX):
            # update its children nodes' parent
            for child_name in cur_node['children']:
                ast_tree[child_name]['parent'] = new_cur_node_ind
        else:
            pass

        # 2. delete old node
        ast_tree.pop(cur_node_ind)

        child_node_indices = util.get_tree_children_func(cur_node)

        if len(child_node_indices) == 0:
            return

        for child_name in child_node_indices:
            dfs(child_name)

    dfs(root_node_ind)

    # recover name
    node_names = deepcopy(list(ast_tree.keys()))
    for node_name in node_names:
        node = deepcopy(ast_tree[node_name])
        if node['children'][0].startswith('_' + constants.NODE_FIX):
            node['children'] = [child_name[1:] for child_name in node['children']]
        else:
            pass
        if node['parent'] == None:
            pass
        else:
            node['parent'] = node['parent'][1:]
        ast_tree[node_name[1:]] = node
        ast_tree.pop(node_name)

    return ast_tree


def parse_base(ast_tree: Dict) -> Dict:
    # delete nodes with single node,eg. [1*NODEFIX1] ->  [1*NODEFIX2] -> ['void'] => [1*NODEFIX1] -> ['void']
    ast_tree = delete_single_child_ndoe(ast_tree)
    ast_tree = binarize_tree(ast_tree)  # to binary ast tree
    ast_tree = reset_indices(ast_tree)  # reset node indices
    return ast_tree
