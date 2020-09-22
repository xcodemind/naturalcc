# -*- coding: utf-8 -*-


import dgl
import torch
from dataset.csn import MAX_SUB_TOKEN_LEN


def tree2graph(tree_dict, dictionary, DGLGraph_PAD_WORD=-1):
    """
    if _subtoken == True, it means that we tokenize leaf node info into sub-tokens
        e.g. ["sub_token", ["sub", "token", <PAD>, <PAD>, <PAD>]]
    else, no tokenization. e.g. ["sub_token"]
    """
    _subtoken = False
    for node in tree_dict.values():
        if isinstance(node['children'][1], list):
            _subtoken = True
            break

    def nonleaf_node_info():
        if _subtoken:
            return [DGLGraph_PAD_WORD] * MAX_SUB_TOKEN_LEN
        else:
            return [DGLGraph_PAD_WORD]

    def token2idx(node_info):
        """
        node info => indices
        if _subtoken == True, ["sub_token", ["sub", "token", <PAD>, <PAD>, <PAD>]] => index(["sub", "token", <PAD>, <PAD>, <PAD>])
        else, ["sub_token"] => index(["sub_token"])
        """
        if _subtoken:
            return [dictionary.index(subtoken) for subtoken in node_info[-1]]
        else:
            return [dictionary.index(node_info[0])]

    """
    how to build DGL graph?
    node: 
        x: node info (if it's non-leaf nodes, padded with [-1, ...]),
        y: current node idx
        mask: if leaf node, mask=1; else, mask=0
        * if current node is the root node,
    edge: child => parent 
    """
    dgl_graph = dgl.DGLGraph()
    ids = sorted(tree_dict.keys(), key=int)

    dgl_graph.add_nodes(
        len(tree_dict),
        data={
            'x': torch.LongTensor([
                token2idx(tree_dict[idx]['children']) if isinstance(tree_dict[idx]['children'][1], list) \
                    else nonleaf_node_info()
                for idx in ids
            ]),
            'y': torch.arange(start=0, end=len(tree_dict)).long(),
            'mask': torch.LongTensor([isinstance(tree_dict[idx]['children'][1], list) for idx in ids]),
        }
    )

    for idx in ids:
        node = tree_dict[idx]
        if node['parent'] is not None:
            dgl_graph.add_edges(int(idx), int(node['parent']))

    return dgl_graph


if __name__ == '__main__':
    from ncc.tasks.summarization import SummarizationTask

    dict = SummarizationTask.load_dictionary(
        filename='/home/yang/.ncc/code_search_net/summarization/data-raw/ruby/bin_ast.dict.json'
    )
    bin_ast = {
        "0": {"type": "method", "parent": None, "children": [1, 2]},
        "1": {"type": "def_keyword", "parent": 0, "children": ["def", ["def", "<pad>", "<pad>", "<pad>", "<pad>"]]},
        "2": {"type": "TMP", "parent": 0, "children": [3, 4]},
        "3": {"type": "identifier", "parent": 2, "children": ["set", ["set", "<pad>", "<pad>", "<pad>", "<pad>"]]},
        "4": {"type": "TMP", "parent": 2, "children": [5, 10]},
        "5": {"type": "method_parameters", "parent": 4, "children": [6, 7]},
        "6": {"type": "LeftParenOp", "parent": 5, "children": ["(", ["(", "<pad>", "<pad>", "<pad>", "<pad>"]]},
        "7": {"type": "TMP", "parent": 5, "children": [8, 9]}, "8": {"type": "identifier", "parent": 7,
                                                                     "children": ["set_attributes",
                                                                                  ["set", "attributes", "<pad>",
                                                                                   "<pad>", "<pad>"]]},
        "9": {"type": "LeftParenOp", "parent": 7, "children": [")", [")", "<pad>", "<pad>", "<pad>", "<pad>"]]},
        "10": {"type": "TMP", "parent": 4, "children": [11, 26]},
        "11": {"type": "assignment", "parent": 10, "children": [12, 13]},
        "12": {"type": "identifier", "parent": 11,
               "children": ["old_attributes", ["old", "attributes", "<pad>", "<pad>", "<pad>"]]},
        "13": {"type": "TMP", "parent": 11, "children": [14, 15]},
        "14": {"type": "AsgnOp", "parent": 13, "children": ["=", ["=", "<pad>", "<pad>", "<pad>", "<pad>"]]},
        "15": {"type": "method_call", "parent": 13, "children": [16, 17]},
        "16": {"type": "identifier", "parent": 15,
               "children": ["compute_attributes", ["compute", "attributes", "<pad>", "<pad>", "<pad>"]]},
        "17": {"type": "argument_list", "parent": 15, "children": [18, 19]},
        "18": {"type": "LeftParenOp", "parent": 17,
               "children": ["(", ["(", "<pad>", "<pad>", "<pad>", "<pad>"]]},
        "19": {"type": "TMP", "parent": 17, "children": [20, 25]},
        "20": {"type": "call", "parent": 19, "children": [21, 22]},
        "21": {"type": "identifier", "parent": 20,
               "children": ["set_attributes", ["set", "attributes", "<pad>", "<pad>", "<pad>"]]},
        "22": {"type": "TMP", "parent": 20, "children": [23, 24]},
        "23": {"type": "DotOp", "parent": 22, "children": [".", [".", "<pad>", "<pad>", "<pad>", "<pad>"]]},
        "24": {"type": "identifier", "parent": 22,
               "children": ["keys", ["keys", "<pad>", "<pad>", "<pad>", "<pad>"]]},
        "25": {"type": "LeftParenOp", "parent": 19,
               "children": [")", [")", "<pad>", "<pad>", "<pad>", "<pad>"]]},
        "26": {"type": "TMP", "parent": 10, "children": [27, 34]},
        "27": {"type": "method_call", "parent": 26, "children": [28, 29]},
        "28": {"type": "identifier", "parent": 27,
               "children": ["assign_attributes", ["assign", "attributes", "<pad>", "<pad>", "<pad>"]]},
        "29": {"type": "argument_list", "parent": 27, "children": [30, 31]},
        "30": {"type": "LeftParenOp", "parent": 29,
               "children": ["(", ["(", "<pad>", "<pad>", "<pad>", "<pad>"]]},
        "31": {"type": "TMP", "parent": 29, "children": [32, 33]},
        "32": {"type": "identifier", "parent": 31,
               "children": ["set_attributes", ["set", "attributes", "<pad>", "<pad>", "<pad>"]]},
        "33": {"type": "LeftParenOp", "parent": 31,
               "children": [")", [")", "<pad>", "<pad>", "<pad>", "<pad>"]]},
        "34": {"type": "TMP", "parent": 26, "children": [35, 36]},
        "35": {"type": "yield_keyword", "parent": 34,
               "children": ["yield", ["yield", "<pad>", "<pad>", "<pad>", "<pad>"]]},
        "36": {"type": "TMP", "parent": 34, "children": [37, 46]},
        "37": {"type": "ensure", "parent": 36, "children": [38, 39]},
        "38": {"type": "ensure_keyword", "parent": 37,
               "children": ["ensure", ["ensure", "<pad>", "<pad>", "<pad>", "<pad>"]]},
        "39": {"type": "method_call", "parent": 37, "children": [40, 41]},
        "40": {"type": "identifier", "parent": 39,
               "children": ["assign_attributes", ["assign", "attributes", "<pad>", "<pad>", "<pad>"]]},
        "41": {"type": "argument_list", "parent": 39, "children": [42, 43]},
        "42": {"type": "LeftParenOp", "parent": 41,
               "children": ["(", ["(", "<pad>", "<pad>", "<pad>", "<pad>"]]},
        "43": {"type": "TMP", "parent": 41, "children": [44, 45]}, "44": {"type": "identifier", "parent": 43,
                                                                          "children": ["old_attributes",
                                                                                       ["old", "attributes",
                                                                                        "<pad>", "<pad>",
                                                                                        "<pad>"]]},
        "45": {"type": "LeftParenOp", "parent": 43,
               "children": [")", [")", "<pad>", "<pad>", "<pad>", "<pad>"]]},
        "46": {"type": "end_keyword", "parent": 36,
               "children": ["end", ["end", "<pad>", "<pad>", "<pad>", "<pad>"]]}}
    tree2graph(bin_ast, dict)
