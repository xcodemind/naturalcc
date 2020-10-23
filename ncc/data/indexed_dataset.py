# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
import os
import shutil
import struct
import numpy as np
import torch
from .ncc_dataset import NccDataset
import ujson
from ncc.utils import tokenizer
import json
from dataset.csn import PATH_NUM
# from ncc.utils.util_graph import (build_graph, tree2graph)
from ncc import LOGGER



def __best_fitting_dtype(vocab_size=None):
    if vocab_size is not None and vocab_size < 65500:
        return np.uint16
    else:
        return np.int32


def get_available_dataset_impl():
    return ['raw', 'lazy', 'cached', 'mmap']


def infer_dataset_impl(path):
    if IndexedRawTextDataset.exists(path):
        return 'raw'
    elif IndexedDataset.exists(path):
        with open(index_file_path(path), 'rb') as f:
            magic = f.read(8)
            if magic == IndexedDataset._HDR_MAGIC:
                return 'cached'
            elif magic == MMapIndexedDataset.Index._HDR_MAGIC[:8]:
                return 'mmap'
            else:
                return None
    else:
        return None


def make_builder(out_file, impl, vocab_size=None):
    if impl == 'mmap':
        return MMapIndexedDatasetBuilder(out_file, dtype=__best_fitting_dtype(vocab_size))
    else:
        return IndexedDatasetBuilder(out_file)


def make_dataset(path, impl, modality='text', fix_lua_indexing=False, dictionary=None, tokenizer=None):
    """
    TODO: should split those IndexedXXDataset into a new file, because with the increasing of development, there must
          be lots of raw dataset load/save rules, and thereofer we should define them in a new py script/directory.
          We should alias IndexedXXDataset into IndexedJsonDataset(load json)/IndexedRawDataset(load string) by
          their loading format function, and an extra mapping function from modality to IndexedXXDataset.
          Examples::
            func = load_dataset(modality='dfs') # func = IndexedJsonDataset (namely, IndexedDFSASTDataset)
            return func(path, dictionary)
    """
    if impl == 'raw' and IndexedRawTextDataset.exists(path):
        if modality == 'path':
            return IndexedRawPathDataset(path, dictionary)
        elif modality == 'bin_ast':
            return IndexedRawASTDataset(path, dictionary)
        elif modality == 'ids':
            return IndexedRawNodeIdDataset(path, dictionary)
        elif modality == 'dfs':
            return IndexedDFSASTDataset(path, dictionary, append_eos=False)
        elif modality in ['tok', 'code', 'code_tokens', 'docstring_tokens', 'ast_tokens']:
            return IndexedTokenDataset(path, dictionary)
        # elif modality == 'javascript_augmented':
        #     return IndexedJavascriptAugmentedDataset(path, dictionary, append_eos=False)
        elif modality == 'javascript':
            return IndexedJavascriptDataset(path, dictionary, append_eos=False)
        elif modality == 'codetype_code':
            return IndexedCodeDataset(path, dictionary, append_eos=False)
        elif modality == 'codetype_type':
            return IndexedTypeDataset(path, dictionary, append_eos=False)
        else:
            assert dictionary is not None
            return IndexedRawTextDataset(path, dictionary)

    elif impl == 'lazy' and IndexedDataset.exists(path):
        return IndexedDataset(path, fix_lua_indexing=fix_lua_indexing)
    elif impl == 'cached' and IndexedDataset.exists(path):
        return IndexedCachedDataset(path, fix_lua_indexing=fix_lua_indexing)
    elif impl == 'mmap' and MMapIndexedDataset.exists(path):
        return MMapIndexedDataset(path)
    elif impl == 'bert':
        return BertDataset(path, dictionary, tokenizer)
    return None


def dataset_exists(path, impl):
    if impl == 'raw':
        return IndexedRawTextDataset.exists(path)
    elif impl == 'mmap':
        return MMapIndexedDataset.exists(path)
    else:
        return IndexedDataset.exists(path)


def read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a


def write_longs(f, a):
    f.write(np.array(a, dtype=np.int64))


dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float,
    7: np.double,
    8: np.uint16
}


def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


def index_file_path(prefix_path):
    return prefix_path + '.idx'


def data_file_path(prefix_path):
    return prefix_path + '.mmap'


class IndexedDataset(NccDataset):
    """Loader for TorchNet IndexedDataset"""
    _HDR_MAGIC = b'TNTIDX\x00\x00'

    def __init__(self, path, fix_lua_indexing=False):
        super().__init__()
        self.path = path
        self.fix_lua_indexing = fix_lua_indexing
        self.data_file = None
        self.read_index(path)

    def read_index(self, path):
        with open(index_file_path(path), 'rb') as f:
            magic = f.read(8)
            assert magic == self._HDR_MAGIC, (
                'Index file doesn\'t match expected format. '
                'Make sure that --dataset-impl is configured properly.'
            )
            version = f.read(8)
            assert struct.unpack('<Q', version) == (1,)
            code, self.element_size = struct.unpack('<QQ', f.read(16))
            self.dtype = dtypes[code]
            self._len, self.s = struct.unpack('<QQ', f.read(16))
            self.dim_offsets = read_longs(f, self._len + 1)
            self.data_offsets = read_longs(f, self._len + 1)
            self.sizes = read_longs(f, self.s)

    def read_data(self, path):
        self.data_file = open(data_file_path(path), 'rb', buffering=0)

    def check_index(self, i):
        if i < 0 or i >= self._len:
            raise IndexError('index out of range')

    def __del__(self):
        if self.data_file:
            self.data_file.close()

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        if not self.data_file:
            self.read_data(self.path)
        self.check_index(i)
        tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
        a = np.empty(tensor_size, dtype=self.dtype)
        self.data_file.seek(self.data_offsets[i] * self.element_size)
        self.data_file.readinto(a)
        item = torch.from_numpy(a).long()
        if self.fix_lua_indexing:
            item -= 1  # subtract 1 for 0-based indexing
        return item

    def __len__(self):
        return self._len

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return (
            os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))
        )

    @property
    def supports_prefetch(self):
        return False  # avoid prefetching to save memory


class IndexedCachedDataset(IndexedDataset):

    def __init__(self, path, fix_lua_indexing=False):
        super().__init__(path, fix_lua_indexing=fix_lua_indexing)
        self.cache = None
        self.cache_index = {}

    @property
    def supports_prefetch(self):
        return True

    def prefetch(self, indices):
        if all(i in self.cache_index for i in indices):
            return
        if not self.data_file:
            self.read_data(self.path)
        indices = sorted(set(indices))
        total_size = 0
        for i in indices:
            total_size += self.data_offsets[i + 1] - self.data_offsets[i]
        self.cache = np.empty(total_size, dtype=self.dtype)
        ptx = 0
        self.cache_index.clear()
        for i in indices:
            self.cache_index[i] = ptx
            size = self.data_offsets[i + 1] - self.data_offsets[i]
            a = self.cache[ptx: ptx + size]
            self.data_file.seek(self.data_offsets[i] * self.element_size)
            self.data_file.readinto(a)
            ptx += size
        if self.data_file:
            # close and delete data file after prefetch so we can pickle
            self.data_file.close()
            self.data_file = None

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
        a = np.empty(tensor_size, dtype=self.dtype)
        ptx = self.cache_index[i]
        np.copyto(a, self.cache[ptx: ptx + a.size])
        item = torch.from_numpy(a).long()
        if self.fix_lua_indexing:
            item -= 1  # subtract 1 for 0-based indexing
        return item


class IndexedRawTextDataset(NccDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, dictionary, append_eos=True, reverse_order=False):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.read_data(path, dictionary)
        self.size = len(self.tokens_list)

    def read_data(self, path, dictionary):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                self.lines.append(line.strip('\n'))
                tokens = dictionary.encode_line(
                    line, tokenizer.tokenize_line, add_if_not_exist=False,
                    append_eos=self.append_eos, reverse_order=self.reverse_order,
                ).long()
                self.tokens_list.append(tokens)
                self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return os.path.exists(path)


class IndexedRawASTDataset(NccDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, dictionary, append_eos=True, reverse_order=False):
        self.tokens_list = []
        self.sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.read_data(path, dictionary)
        self.size = len(self.tokens_list)

    def read_data(self, path, dictionary):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line.strip('\n'))
                graph = build_graph(line, dictionary)
                self.tokens_list.append(graph)
                self.sizes.append(len(graph))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        raise NotImplementedError

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return os.path.exists(path)


class IndexedRawNodeIdDataset(NccDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        # self.append_eos = append_eos
        # self.reverse_order = reverse_order
        self.read_data(path)
        self.size = len(self.tokens_list)

    def read_data(self, path, dictionary):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line.strip('\n'))
                self.lines.append(line)
                tokens = dictionary.encode_ast(
                    line, add_if_not_exist=False,
                    append_eos=self.append_eos, reverse_order=self.reverse_order,
                ).long()
                self.tokens_list.append(tokens)
                self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return os.path.exists(path)


class IndexedDFSASTDataset(NccDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, dictionary, append_eos=True, reverse_order=False):
        self.tokens_list = []
        self.lines = []
        self.extends = []
        self.sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.read_data(path, dictionary)
        self.size = len(self.tokens_list)

    def read_data(self, path, dictionary):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = ujson.loads(line.strip('\n'))
                line, ext = line[:-1], line[-1]
                self.lines.append(line)
                self.extends.append(ext)
                tokens = dictionary.encode_tok(
                    line, add_if_not_exist=False,
                    append_eos=self.append_eos, reverse_order=self.reverse_order,
                ).long()
                self.tokens_list.append(tokens)
                self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return os.path.exists(path)


class IndexedRawPathDataset(NccDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""
    PATH_LENGTH = PATH_NUM * 3

    def __init__(self, path, dictionary, append_eos=False, reverse_order=False):
        # self.lines = []# because 1) path modality data are too large adnd 2) we cannot understand code via path data
        self.tokens_list = []
        self.sizes = []  # body size
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.read_data(path, dictionary)
        self.size = len(self.tokens_list)

    def read_data(self, path, dictionary):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = ujson.loads(line)
                line = dictionary.encode_line(line, line_tokenizer=None, add_if_not_exist=False,
                                              append_eos=self.append_eos, reverse_order=self.reverse_order).long()
                self.tokens_list.append(line)
                self.sizes.append(len(line))

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        raise NotImplementedError

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return os.path.exists(path)


class IndexedTokenDataset(NccDataset):
    def __init__(self, path, dictionary, append_eos=True, reverse_order=False):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.read_data(path, dictionary)
        self.size = len(self.tokens_list)

    def read_data(self, path, dictionary):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = ujson.loads(line)
                self.lines.append(line)
                tokens = dictionary.encode_tok(
                    line, add_if_not_exist=False,
                    append_eos=self.append_eos, reverse_order=self.reverse_order,
                ).long()
                self.tokens_list.append(tokens)
                self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return os.path.exists(path)


class IndexedJavascriptDataset(NccDataset):
    def __init__(self, path, dictionary, append_eos=False, reverse_order=False):
        self.examples_list = []
        self.lines = []
        self.sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.read_data(path, dictionary)
        self.size = len(self.examples_list)

    def read_data(self, path, dictionary, max_source_positions=1024):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                program = ujson.loads(line)
                # programs = []
                # for program in line:
                # TODO
                # program = [dictionary.bos_word] + program[: max_source_positions-2] + [dictionary.eos_word]
                program = dictionary.encode_line(program, line_tokenizer=None, add_if_not_exist=False,
                                                 append_eos=self.append_eos, reverse_order=self.reverse_order).long()
                # programs.append(program)

                self.examples_list.append(program)
                self.sizes.append(len(program))

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.examples_list[i]

    # def get_original_text(self, i):
    #     self.check_index(i)
    #     return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return os.path.exists(path)

# Has been moved to tasks/contracode_hybrid.py
# class IndexedJavascriptAugmentedDataset(NccDataset):
#     def __init__(self, path, dictionary, append_eos=False, reverse_order=False):
#         self.examples_list = []
#         self.lines = []
#         self.sizes = []
#         self.append_eos = append_eos
#         self.reverse_order = reverse_order
#         self.read_data(path, dictionary)
#         self.size = len(self.examples_list)
#
#     def read_data(self, path, dictionary, max_source_positions=1024):
#         with open(path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 line = ujson.loads(line)
#                 programs = []
#                 for program in line:
#                     # TODO
#                     program = [dictionary.bos_word] + program[: max_source_positions - 2] + [dictionary.eos_word]
#                     program = dictionary.encode_line(program, line_tokenizer=None, add_if_not_exist=False,
#                                                      append_eos=self.append_eos,
#                                                      reverse_order=self.reverse_order).long()
#                     programs.append(program)
#
#                 self.examples_list.append(programs)
#                 self.sizes.append(len(programs[0]))
#
#     def check_index(self, i):
#         if i < 0 or i >= self.size:
#             raise IndexError('index out of range')
#
#     @lru_cache(maxsize=8)
#     def __getitem__(self, i):
#         self.check_index(i)
#         return self.examples_list[i]
#
#     # def get_original_text(self, i):
#     #     self.check_index(i)
#     #     return self.lines[i]
#
#     def __del__(self):
#         pass
#
#     def __len__(self):
#         return self.size
#
#     def num_tokens(self, index):
#         return self.sizes[index]
#
#     def size(self, index):
#         return self.sizes[index]
#
#     @staticmethod
#     def exists(path):
#         return os.path.exists(path)
#

class IndexedCodeDataset(NccDataset):
    def __init__(self, path, dictionary, append_eos=False, reverse_order=False):
        self.examples_list = []
        self.lines = []
        self.sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.read_data(path, dictionary)
        self.size = len(self.examples_list)

    def read_data(self, path, dictionary, max_source_positions=1024):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                program = line.rstrip()
                self.lines.append(line)
                self.examples_list.append(program)
                self.sizes.append(len(program.split(' ')))  # TODO: simply calculate the length by space tokenization

        # with open(path, 'r', encoding='utf-8') as f:
        #     for line in f:
        #         program = ujson.loads(line)
        #         # programs = []
        #         # for program in line:
        #         # TODO
        #         program = [dictionary.bos_word] + program[: max_source_positions-2] + [dictionary.eos_word]
        #         program = dictionary.encode_line(program, line_tokenizer=None, add_if_not_exist=False,
        #                                       append_eos=self.append_eos, reverse_order=self.reverse_order).long()
        #         # programs.append(program)
        #
        #         self.examples_list.append(program)
        #         self.sizes.append(len(program))

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.examples_list[i]

    # def get_original_text(self, i):
    #     self.check_index(i)
    #     return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return os.path.exists(path)


class IndexedTypeDataset(NccDataset):
    def __init__(self, path, dictionary, append_eos=False, reverse_order=False):
        self.examples_list = []
        self.lines = []
        self.sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.read_data(path, dictionary)
        self.size = len(self.examples_list)

    def read_data(self, path, dictionary, max_source_positions=1024):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                type = line.rstrip().strip('\n')
                self.lines.append(line)
                self.examples_list.append(type)
                self.sizes.append(len(type.split(' ')))

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.examples_list[i]

    # def get_original_text(self, i):
    #     self.check_index(i)
    #     return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return os.path.exists(path)


class BertDataset(NccDataset):
    def __init__(self, path, dictionary, tokenizer):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        # self.append_eos = append_eos
        # self.reverse_order = reverse_order
        self.read_data(path, dictionary, tokenizer)
        self.size = len(self.tokens_list)

    def read_data(self, path, dictionary, tokenizer, block_size=512):
        # tokenizer = ByteLevelBPETokenizer(
        #     os.path.join(self.args['dataset']['tokenizer_name'], 'vocab.json'),
        #     os.path.join(self.args['dataset']['tokenizer_name'], 'merges.txt'),
        # )
        # tokenizer._tokenizer.post_processor = BertProcessing(
        #     ("</s>", tokenizer.token_to_id("</s>")),
        #     ("<s>", tokenizer.token_to_id("<s>")),
        # )
        # tokenizer.enable_truncation(max_length=512)

        # or use the RobertaTokenizer from `transformers` directly.
        with open(path, 'r', encoding='utf-8') as f:
            # for line in f:
            #     self.lines.append(line.strip('\n'))
            #     tokens = dictionary.encode_line(
            #         line, add_if_not_exist=False,
            #         append_eos=self.append_eos, reverse_order=self.reverse_order,
            #     ).long()
            #     self.tokens_list.append(tokens)
            #     self.sizes.append(len(tokens))

            # text = f.read()
            # tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            self.lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

            # for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
            #     self.tokens_list.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i: i + block_size]))
            # self.sizes = np.array(self.sizes)
        self.tokens_list = tokenizer.batch_encode_plus(self.lines, add_special_tokens=True, max_length=block_size)[
            "input_ids"]
        self.sizes = np.array([len(tok) for tok in self.tokens_list])
        self.tokens_list = [torch.LongTensor(tok) for tok in self.tokens_list]

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return os.path.exists(path)


class IndexedDatasetBuilder(object):
    element_sizes = {
        np.uint8: 1,
        np.int8: 1,
        np.int16: 2,
        np.int32: 4,
        np.int64: 8,
        np.float: 4,
        np.double: 8
    }

    def __init__(self, out_file, dtype=np.int32):
        self.out_file = open(out_file, 'wb')
        self.dtype = dtype
        self.data_offsets = [0]
        self.dim_offsets = [0]
        self.sizes = []
        self.element_size = self.element_sizes[self.dtype]

    def add_item(self, tensor):
        # +1 for Lua compatibility
        bytes = self.out_file.write(np.array(tensor.numpy() + 1, dtype=self.dtype))
        self.data_offsets.append(self.data_offsets[-1] + bytes / self.element_size)
        for s in tensor.size():
            self.sizes.append(s)
        self.dim_offsets.append(self.dim_offsets[-1] + len(tensor.size()))

    def merge_file_(self, another_file):
        index = IndexedDataset(another_file)
        assert index.dtype == self.dtype

        begin = self.data_offsets[-1]
        for offset in index.data_offsets[1:]:
            self.data_offsets.append(begin + offset)
        self.sizes.extend(index.sizes)
        begin = self.dim_offsets[-1]
        for dim_offset in index.dim_offsets[1:]:
            self.dim_offsets.append(begin + dim_offset)

        with open(data_file_path(another_file), 'rb') as f:
            while True:
                data = f.read(1024)
                if data:
                    self.out_file.write(data)
                else:
                    break

    def finalize(self, index_file):
        self.out_file.close()
        index = open(index_file, 'wb')
        index.write(b'TNTIDX\x00\x00')
        index.write(struct.pack('<Q', 1))
        index.write(struct.pack('<QQ', code(self.dtype), self.element_size))
        index.write(struct.pack('<QQ', len(self.data_offsets) - 1, len(self.sizes)))
        write_longs(index, self.dim_offsets)
        write_longs(index, self.data_offsets)
        write_longs(index, self.sizes)
        index.close()


def _warmup_mmap_file(path):
    with open(path, 'rb') as stream:
        while stream.read(100 * 1024 * 1024):
            pass


class MMapIndexedDataset(torch.utils.data.Dataset):
    class Index(object):
        _HDR_MAGIC = b'MMIDIDX\x00\x00'

        @classmethod
        def writer(cls, path, dtype):
            class _Writer(object):
                def __enter__(self):
                    """for with open. this init method"""
                    self._file = open(path, 'wb')

                    self._file.write(cls._HDR_MAGIC)  # self-defined format
                    self._file.write(struct.pack('<Q', 1))  # version number, occupying 8 bit
                    self._file.write(struct.pack('<B', code(dtype)))  # data type, 1 bit

                    return self

                @staticmethod
                def _get_pointers(sizes):
                    dtype_size = dtype().itemsize
                    address = 0
                    pointers = []

                    for size in sizes:
                        pointers.append(address)
                        address += size * dtype_size

                    return pointers

                def write(self, sizes):
                    pointers = self._get_pointers(sizes)

                    self._file.write(struct.pack('<Q', len(sizes)))

                    sizes = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes.tobytes(order='C'))
                    del sizes

                    pointers = np.array(pointers, dtype=np.int64)
                    self._file.write(pointers.tobytes(order='C'))
                    del pointers

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path):
            with open(path, 'rb') as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    'Index file doesn\'t match expected format. '
                    'Make sure that --dataset-impl is configured properly.'
                )
                version = struct.unpack('<Q', stream.read(8))
                assert (1,) == version

                dtype_code, = struct.unpack('<B', stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack('<Q', stream.read(8))[0]
                offset = stream.tell()

            _warmup_mmap_file(path)

            self._bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            self._sizes = np.frombuffer(self._bin_buffer, dtype=np.int32, count=self._len, offset=offset)
            self._pointers = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._len,
                                           offset=offset + self._sizes.nbytes)

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def _do_init(self, path):
        self._path = path
        self._index = self.Index(index_file_path(self._path))

        _warmup_mmap_file(data_file_path(self._path))
        self._bin_buffer_mmap = np.memmap(data_file_path(self._path), mode='r', order='C')
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        ptr, size = self._index[i]
        np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr)
        if self._index.dtype != np.int64:
            np_array = np_array.astype(np.int64)

        return torch.from_numpy(np_array)

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return (
            os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))
        )


class MMapIndexedDatasetBuilder(object):
    """memory-mapping dataset builder with index"""

    def __init__(self, out_file, dtype=np.int64):
        self._data_file = open(out_file, 'wb')
        self._dtype = dtype
        self._sizes = []

    def add_item(self, tensor):
        # write an array
        np_array = np.array(tensor.numpy(), dtype=self._dtype)  # type transform
        # bin file
        self._data_file.write(np_array.tobytes(order='C'))  # write np.array into C stream
        # idx file
        self._sizes.append(np_array.size)

    def merge_file_(self, another_file):
        """merge sub file(bin/idx) for multi-processing"""
        # Concatenate index
        index = MMapIndexedDataset.Index(index_file_path(another_file))
        assert index.dtype == self._dtype

        for size in index.sizes:
            self._sizes.append(size)

        # Concatenate data
        with open(data_file_path(another_file), 'rb') as f:
            # append sub-bin/idx files to 1st bin/idx file
            shutil.copyfileobj(f, self._data_file)

    def finalize(self, index_file):
        # assert len(self._sizes) > 0, Exception('{} {}'.format(self._data_file, self._sizes))
        self._data_file.close()

        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes)
