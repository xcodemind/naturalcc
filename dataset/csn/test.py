# -*- coding: utf-8 -*-

# import ujson
#
# with open('/home/yang/.ncc/code_search_net/flatten/ruby/train.raw_ast', 'r') as reader:
#     for idx, line in enumerate(reader):
#         ast = ujson.loads(line)
#         if len(ast) > 5000:
#             print(idx, len(ast))

from dataset.csn.utils.util_path import ast_to_path

paths = ast_to_path(ast_list[0], MAX_PATH=300)
paths
