# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
import ujson

SPACE_NORMALIZER = re.compile(r"\s+")
PAD_WORD = '<PAD>'

def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def tokenize_tree_line(line):
    line = ujson.loads(line)
    leaf_node_tokens = []
    for _, node_info in line.items():
        if type(node_info['children'][-1]) == list:
            for token in node_info['children'][-1]:
                if token != PAD_WORD:
                    leaf_node_tokens.append(token)
                else:
                    break

    return leaf_node_tokens


def tokenize_path_line(line):
    line = ujson.loads(line)
    border_list, center_list = [], []
    for path in line:
        head, center, tail = path
        border_list.extend(head + tail)
        center_list.extend(center)

    return border_list, center_list


# def list_to_dict(dst_dir: str, dict_filename: str, min_freq=2):
#     train_files = [file for file in glob('{}/*'.format(dst_dir)) if 'train' in file.split('/')[-1]]
#     train_counters = []
#     list_buffer = []
#     for file in train_files:
#         with open(file, 'r') as reader:
#             line = reader.readline().strip()
#             while len(line) > 0:
#                 line = ujson.loads(line)
#                 list_buffer.append(line)
#
#                 if len(list_buffer) >= 5000:
#                     train_counters.append(Counter(itertools.chain(*list_buffer)))
#                     list_buffer = []
#
#                 line = reader.readline().strip()
#     if len(list_buffer) > 0:
#         train_counters.append(Counter(itertools.chain(*list_buffer)))
#         list_buffer = []
#
#     token_dict = merge_counters(train_counters)
#     tokens_dict = {key: freq for key, freq in token_dict.items() if freq > min_freq}
#     sorted_tokens = sort_by_freq(tokens_dict)
#     dump_dict(sorted_tokens, dict_filename)
#
#
# def tree_to_dict(dst_dir: str, dict_filename: str, min_freq=2):
#     train_files = [file for file in glob('{}/*'.format(dst_dir)) if 'train' in file.split('/')[-1]]
#     train_counters = []
#     list_buffer = []
#     for file in train_files:
#         with open(file, 'r') as reader:
#             line = reader.readline().strip()
#             while len(line) > 0:
#                 line = ujson.loads(line)
#
#                 leaf_node_tokens = []
#                 for node_inf, node_info in line.items():
#                     if type(node_info['children'][-1]) == list:
#                         for token in node_info['children'][-1]:
#                             if token != PAD_WORD:
#                                 leaf_node_tokens.append(token)
#                             else:
#                                 break
#                 list_buffer.append(leaf_node_tokens)
#
#                 if len(list_buffer) >= 5000:
#                     train_counters.append(Counter(itertools.chain(*list_buffer)))
#                     list_buffer = []
#                 line = reader.readline().strip()
#     if len(list_buffer) > 0:
#         train_counters.append(Counter(itertools.chain(*list_buffer)))
#         list_buffer = []
#
#     token_dict = merge_counters(train_counters)
#     tokens_dict = {key: freq for key, freq in token_dict.items() if freq > min_freq}
#     sorted_tokens = sort_by_freq(tokens_dict)
#     dump_dict(sorted_tokens, dict_filename)
#
#
# def path_to_dict(dst_dir: str, border_dict_filename: str, center_dict_filename: str, min_freq=2):
#     train_files = [file for file in glob('{}/*'.format(dst_dir)) if 'train' in file.split('/')[-1]]
#     border_counters, center_counters = [], []
#     border_list_buffer, center_list_buffer = [], []
#     for file in train_files:
#         with open(file, 'r') as reader:
#             line = reader.readline().strip()
#             while len(line) > 0:
#                 line = ujson.loads(line)
#
#                 for path in line:
#                     head, center, tail = path
#                     border_list_buffer.append(head + tail)
#                     center_list_buffer.append(center)
#
#                 if len(border_list_buffer) >= 5000 or len(center_list_buffer) > 5000:
#                     border_counters.append(Counter(itertools.chain(*border_list_buffer)))
#                     center_counters.append(Counter(itertools.chain(*center_list_buffer)))
#                     border_list_buffer, center_list_buffer = [], []
#                 line = reader.readline().strip()
#     if len(border_list_buffer) >= 0 or len(center_list_buffer) > 0:
#         border_counters.append(Counter(itertools.chain(*border_list_buffer)))
#         center_counters.append(Counter(itertools.chain(*center_list_buffer)))
#         border_list_buffer, center_list_buffer = [], []
#
#     border_dict = merge_counters(border_counters)
#     border_dict = {key: freq for key, freq in border_dict.items() if freq > min_freq}
#     sorted_border_tokens = sort_by_freq(border_dict)
#     dump_dict(sorted_border_tokens, border_dict_filename)
#
#     center_dict = merge_counters(center_counters)
#     center_dict = {key: freq for key, freq in center_dict.items() if freq > min_freq}
#     sorted_center_tokens = sort_by_freq(center_dict)
#     dump_dict(sorted_center_tokens, center_dict_filename)

