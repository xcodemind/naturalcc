# -*- coding: utf-8 -*-

def get_dfs(ast, only_leaf=False):
    """get token(namely, type node or value node) list of a ast"""
    dp = []
    for node in ast:
        if "value" in node:
            dp.append(node["value"])
        else:
            if not only_leaf:
                dp.append(node["type"])
    return dp


def separate_dps(ast, max_len):
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


def get_rel_masks(dp, max_len):
    """get trav_trans mask"""

    def get_ancestors(dp):
        ancestors = {0: []}
        node2parent = {0: 0}
        levels = {0: 0}
        for i, node in enumerate(dp):
            if "children" in node:
                cur_level = levels[i]
                for child in node["children"]:
                    node2parent[child] = i
                    levels[child] = cur_level + 1
            ancestors[i] = [i] + ancestors[node2parent[i]]
        return ancestors, levels

    def get_path(i, j, anc_i, j_level):
        if i - j >= max_len:
            return '0'
        for node in ancestors[j][j_level:]:
            if node in anc_i:
                up_n = levels[i] - levels[node]
                down_n = levels[j] - levels[node]
                return '{}|{}'.format(up_n, down_n)  # 50 samples, time-consuming: 16.52610754966736

    # ancestors, the path from current node to the root node
    # levels, the depth of current node to the root node
    ancestors, levels = get_ancestors(dp)
    # add an empty list ([]) at the head of path_rels so that we can use authors-defined code
    path_rels = [None] * len(dp)
    for i in range(1, len(dp)):
        anc_i = set(ancestors[i])
        j_level = -(levels[i] + 1)
        path_rels[i] = [get_path(i, j, anc_i, j_level) for j in range(i)]
    return path_rels


def separate_rel_mask(rel_mask, max_len):
    """
    Separate the mask by a sliding window to keep each dp at length max_len.
    For the masks, for each row, since we want the information to be relative
    to whatever is being predicted (ie. input_seq[i+1]), we are shifting
    everything by 1. Thus, the length of each mask will be len(seq) - 1.
    (0)1-max_len, max_len/2-max_len/2+max_len, ...
    """
    if len(rel_mask) <= max_len:
        # return [[" ".join(lst.split()[:-1]) for lst in rel_mask[1:]]]
        return [rel_mask[1:]]

    half_len = int(max_len / 2)
    # rel_mask_aug = [[" ".join(lst.split()[:-1]) for lst in rel_mask[1:max_len]]]
    rel_mask_aug = [rel_mask[1:max_len]]

    i = half_len
    while i < len(rel_mask) - max_len:
        rel_mask_aug.append(
            # [" ".join(lst.split()[i:-1]) for lst in rel_mask[i + 1: i + max_len]]
            [lst[i:] for lst in rel_mask[i + 1: i + max_len]]
        )
        i += half_len
    rel_mask_aug.append(
        [
            lst[-(i + 2): -1]
            for i, lst in enumerate(rel_mask[-max_len + 1:])
        ]
    )
    return rel_mask_aug


def get_leaf_ids(ast):
    ids = {"leaf_ids": [], "internal_ids": []}
    for i, node in enumerate(ast):
        if "value" in node:
            ids["leaf_ids"].append(i)
        else:
            ids["internal_ids"].append(i)
    return ids


def get_value_ids(ast):
    ids = {"attr_ids": [], "num_ids": [], "name_ids": [], "param_ids": []}
    for i, node in enumerate(ast):
        if "type" in node:
            if node["type"] == "attr":
                ids["attr_ids"].append(
                    i + 1
                )  # + 1 since i is the type, and we want the value
            elif node["type"] == "Num":
                ids["num_ids"].append(i + 1)
            elif node["type"] in {"NameLoad", "NameStore"}:
                ids["name_ids"].append(i + 1)
            elif node["type"] == "NameParam":
                ids["param_ids"].append(i + 1)
    return ids


def get_type_ids(ast):
    ids = {
        "call_ids": [],
        "assign_ids": [],
        "return_ids": [],
        "list_ids": [],
        "dict_ids": [],
        "raise_ids": [],
    }
    for i, node in enumerate(ast):
        if "type" in node:
            type_ = node["type"]
            if type_ == "Call":
                ids["call_ids"].append(i)
            elif type_ == "Assign":
                ids["assign_ids"].append(i)
            elif type_ == "Return":
                ids["return_ids"].append(i)
            elif type_ in {"ListComp", "ListLoad", "ListStore"}:
                ids["list_ids"].append(i)
            elif type_ in {"DictComp", "DictLoad", "DictStore"}:
                ids["dict_ids"].append(i)
            elif type_ == "Raise":
                ids["raise_ids"].append(i)
    return ids


IDS_CLS = {
    "leaf_ids", "internal_ids",  # leaf ids
    "attr_ids", "num_ids", "name_ids", "param_ids",  # value ids
    "call_ids", "assign_ids", "return_ids", "list_ids", "dict_ids", "raise_ids",  # type ids
}

__all__ = (
    # dfs ast
    'get_dfs',
    'separate_dps',

    # relative mask
    'get_rel_masks',
    'separate_rel_mask',

    # ids
    'get_leaf_ids',
    'get_value_ids',
    'get_type_ids',

    'IDS_CLS'
)
