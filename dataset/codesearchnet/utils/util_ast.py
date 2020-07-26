# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Optional

import sys
from copy import deepcopy

from dataset.codesearchnet.utils import constants
from dataset.codesearchnet.utils import util
from ncc.data.constants import PAD

# ignore those ast whose size is too large. Therefore set it as a small number
sys.setrecursionlimit(constants.RECURSION_DEPTH)  # recursion depth


def delete_comment_node(ast_tree: Dict) -> Dict:
    '''delete comment node and its children'''

    def delete_cur_node(node_idx, cur_node):
        # update its parent's children
        parent_idx = cur_node['parent']
        parent_node = ast_tree[parent_idx]
        del_idx = parent_node['children'].index(node_idx)
        del parent_node['children'][del_idx]
        # delete node
        ast_tree.pop(node_idx)
        return parent_idx, parent_node

    def dfs(node_idx):
        cur_node = ast_tree[node_idx]
        child_ids = cur_node.get('children', None)

        if 'comment' in cur_node['type']:
            node_idx, cur_node = delete_cur_node(node_idx, cur_node)
            while len(cur_node['children']) == 0:
                node_idx, cur_node = delete_cur_node(node_idx, cur_node)

        if child_ids is None:
            return

        for idx in child_ids:
            dfs(node_idx=idx)

    dfs(node_idx=0)
    return ast_tree


def remove_only_one_child_root(ast_tree: Dict) -> Dict:
    # 3) pop head node which has only 1 child
    # because in such way, head node might be Program/Function/Error and its child is the code's AST
    node_ids = deepcopy(list(ast_tree.keys()))
    for idx in node_ids:
        if (ast_tree[idx]['parent'] is None) and len(ast_tree[idx]['children']) == 1:
            child_idx = ast_tree[idx]['children'][0]
            ast_tree[child_idx]['parent'] = None
            ast_tree.pop(idx)
        else:
            break
    return ast_tree


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


def pad_leaf_node(ast_tree: Dict, MAX_LEN: int, to_lower: bool = True, PAD_TOKEN: str = PAD) -> Dict:
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


def parse_deepcom(ast_tree: dict, sbt_func: Any, to_lower: bool) -> Optional[List]:
    try:
        sbt_seq = sbt_func(ast_tree, constants.ROOT_NODE_NAME, to_lower)
        return sbt_seq
    except Exception as err:
        print(err)
        print(ast_tree)
        return None


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
    if sorted(list(ast_tree.keys())) == list(range(len(ast_tree))):
        return ast_tree

    root_idx = 0
    while 1:
        if root_idx in ast_tree:
            break
        else:
            root_idx += 1  # root node has been removed

    # firstly, resort node index with _
    new_ast_idx = 0

    def dfs(idx):
        nonlocal new_ast_idx
        new_cur_idx, new_ast_idx = '_{}'.format(new_ast_idx), new_ast_idx + 1  # update for next node
        cur_node = ast_tree[idx]
        ast_tree[new_cur_idx] = deepcopy(cur_node)

        # update its parent's children
        if cur_node['parent'] is None:
            pass  # current node is root node, no need for update its children
        else:
            parent_node = ast_tree[cur_node['parent']]
            parent_node['children'][parent_node['children'].index(idx)] = new_cur_idx

        if 'children' in cur_node:
            # update its children nodes' parent
            for child_idx in cur_node['children']:
                ast_tree[child_idx]['parent'] = new_cur_idx
        else:
            pass  # current node is leaf, no children item, only value item

        # 2. delete old node
        ast_tree.pop(idx)

        child_ids = cur_node.get('children', None)

        if child_ids is None:
            return

        for child_idx in child_ids:
            dfs(child_idx)

    dfs(root_idx)

    # recover name: from _* => *
    node_ids = deepcopy(list(ast_tree.keys()))
    for idx in node_ids:
        node = deepcopy(ast_tree[idx])
        if 'children' in node:
            node['children'] = [int(child_idx[1:]) for child_idx in node['children']]
        else:
            pass
        if node['parent'] == None:
            pass
        else:
            node['parent'] = int(node['parent'][1:])
        ast_tree[int(idx[1:])] = node
        ast_tree.pop(idx)

    return ast_tree


def parse_base(ast_tree: Dict) -> Optional[Dict]:
    try:
        # delete nodes with single node,eg. [1*NODEFIX1] ->  [1*NODEFIX2] -> ['void'] => [1*NODEFIX1] -> ['void']
        ast_tree = delete_single_child_ndoe(ast_tree)
        ast_tree = binarize_tree(ast_tree)  # to binary ast tree
        ast_tree = reset_indices(ast_tree)  # reset node indices
        return ast_tree
    except Exception as err:
        print(err)
        print(ast_tree)
        return None


def convert(ast: Dict[int, Dict]) -> List[Dict]:
    new_ast = []
    for idx in deepcopy(sorted(ast.keys(), key=int)):
        node = ast[idx]
        node.pop('parent')
        new_ast.append(node)
    ast = new_ast

    increase_by = {}  # count of how many idx to increase the new idx by:
    # each time there is a value node
    cur = 0
    for i, node in enumerate(ast):
        increase_by[i] = cur
        if "value" in node:
            cur += 1

    new_dp = []
    for i, node in enumerate(ast):
        inc = increase_by[i]
        if "value" in node:
            child = [i + inc + 1]
            if "children" in node:
                child += [n + increase_by[n] for n in node["children"]]
            new_dp.append({"type": node["type"], "children": child})
            new_dp.append({"value": node["value"]})
        else:
            if "children" in node:
                node["children"] = [n + increase_by[n] for n in node["children"]]
            new_dp.append(node)

    # sanity check
    children = []
    for node in new_dp:
        if "children" in node:
            children += node["children"]
    assert len(children) == len(set(children))
    return new_dp


def dfs_traversal(ast: List[Dict], only_leaf=False):
    dfs_seq = []
    for node in ast:
        if "value" in node:
            dfs_seq.append(node["value"])
        else:
            if not only_leaf:
                dfs_seq.append(node["type"])
    return dfs_seq


def separate_ast(ast: List[Dict], max_len: int):
    """
    Handles training / evaluation on long ASTs by splitting
    them into smaller ASTs of length max_len, with a sliding
    window of max_len / 2.

    Example: for an AST ast with length 1700, and max_len = 1000,
    the output will be:
    [[ast[0:1000], 0], [ast[500:1500], 1000], [ast[700:1700], 1500]]

    Input:
        ast : List[Dictionary]
            List of nodes in pre-order traversal.
        max_len : int

    Output:
        aug_asts : List[List[List, int]]
            List of (ast, beginning idx of unseen nodes)
    """
    half_len = int(max_len / 2)
    if len(ast) <= max_len:
        return [[ast, 0]]

    aug_asts = [[ast[:max_len], 0]]
    i = half_len
    while i < len(ast) - max_len:
        aug_asts.append([ast[i: i + max_len], half_len])
        i += half_len
    idx = max_len - (len(ast) - (i + half_len))
    aug_asts.append([ast[-max_len:], idx])
    return aug_asts


# def traversal(ast: Dict, method: str = 'dfs'):
#     if method.lower() == 'dfs':
#
#     else:
#         raise NotImplementedError


if __name__ == '__main__':
    convert(ast)
