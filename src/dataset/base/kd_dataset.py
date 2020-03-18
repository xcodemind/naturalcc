# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

# from src import *
# from src.data import *
#
# import glob
# import random
# from joblib import Parallel, delayed
# from multiprocessing import cpu_count
# import ujson
# import math
#
# from src.utils.util import *
# from src.utils.util_data import *
# from src.utils.util_file import *
# from src.data import _Dict

import itertools
# import torch
from src.utils.util_data import *
# from src.data.codesum.util_data import pad_seq,pad_comment
# from src.data.tree import get_tree_dgl_graph_batch, get_tree_padding_mask
from src.data.dict import Dict
# import src.data.codesum.constants as constants
import struct
import os
# import json
import numpy as np
import bisect
from  src.utils.constants import * 
from  src import LOGGER

dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float,
    7: np.double,
}

def index_file_path(prefix_path):
    return prefix_path + '.idx'

def read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a

def data_file_path(prefix_path):
    return prefix_path + '.bin'

def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k

def write_longs(f, a):
    f.write(np.array(a, dtype=np.int64))

class FairseqDataset(torch.utils.data.Dataset):
    """A dataset that provides helpers for batching."""

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[int]): sample indices to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        raise NotImplementedError

    def get_dummy_batch(self, num_tokens, max_positions):
        """Return a dummy batch with a given number of tokens."""
        raise NotImplementedError

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        raise NotImplementedError

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        raise NotImplementedError

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        raise NotImplementedError

    @property
    def supports_prefetch(self):
        return False

    def prefetch(self, indices):
        raise NotImplementedError

class ConcatDataset(FairseqDataset):

    @staticmethod
    def cumsum(sequence, sample_ratios):
        r, s = [], 0
        for e, ratio in zip(sequence, sample_ratios):
            l = ratio * len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets, sample_ratios=1):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        if isinstance(sample_ratios, int):
            sample_ratios = [sample_ratios] * len(self.datasets)
        self.sample_ratios = sample_ratios
        self.cummulative_sizes = self.cumsum(self.datasets, sample_ratios)
        self.real_sizes = [len(d) for d in self.datasets]

    def __len__(self):
        return self.cummulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cummulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cummulative_sizes[dataset_idx - 1]
        sample_idx = sample_idx % self.real_sizes[dataset_idx]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def sizes(self):
        return np.concatenate([np.tile(ds.sizes, sr) for ds, sr in zip(self.datasets, self.sample_ratios)])

    @property
    def supports_prefetch(self):
        return all([d.supports_prefetch for d in self.datasets])

    def prefetch(self, indices):
        frm = 0
        for to, ds in zip(self.cummulative_sizes, self.datasets):
            real_size = len(ds)
            ds.prefetch([(i - frm) % real_size for i in indices if frm <= i < to])
            frm = to

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
        index = IndexedDataset(another_file, read_data=False)
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

class TeacherOutputDatasetBuilder(IndexedDatasetBuilder):
    def add_item(self, data):
        # +1 for Lua compatibility
        data = np.array(data, dtype=self.dtype)
        bytes = self.out_file.write(data)
        self.data_offsets.append(self.data_offsets[-1] + bytes / self.element_size)
        for s in data.shape:
            self.sizes.append(s)
        self.dim_offsets.append(self.dim_offsets[-1] + len(data.shape))

class IndexedDataset(torch.utils.data.Dataset):
    """Loader for TorchNet IndexedDataset"""

    def __init__(self, path, fix_lua_indexing=False, read_data=True):
        super().__init__()
        self.fix_lua_indexing = fix_lua_indexing
        self.read_index(path)
        self.data_file = None
        if read_data:
            self.read_data(path)

    def read_index(self, path):
        with open(index_file_path(path), 'rb') as f:
            magic = f.read(8)
            assert magic == b'TNTIDX\x00\x00'
            version = f.read(8)
            assert struct.unpack('<Q', version) == (1,)
            code, self.element_size = struct.unpack('<QQ', f.read(16))
            self.dtype = dtypes[code]
            self.size, self.s = struct.unpack('<QQ', f.read(16))
            self.dim_offsets = read_longs(f, self.size + 1)
            self.data_offsets = read_longs(f, self.size + 1)
            self.sizes = read_longs(f, self.s)

    def read_data(self, path):
        self.data_file = open(data_file_path(path), 'rb', buffering=0)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    def __del__(self):
        if self.data_file:
            self.data_file.close()

    def __getitem__(self, i):
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
        return self.size

    @staticmethod
    def exists(path):
        return (
                os.path.exists(index_file_path(path)) and
                os.path.exists(data_file_path(path))
        )

class IndexedCachedDataset(IndexedDataset):

    def __init__(self, path, fix_lua_indexing=False):
        super().__init__(path, fix_lua_indexing, True)
        self.cache = {}

    @property
    def supports_prefetch(self):
        return True

    def prefetch(self, indices):
        pass

    def __getitem__(self, i):
        self.check_index(i)
        tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
        a = np.empty(tensor_size, dtype=self.dtype)

        if i in self.cache:
            np.copyto(a, self.cache[i])
        else:
            self.data_file.seek(self.data_offsets[i] * self.element_size)
            self.data_file.readinto(a)
            self.cache[i] = a

        item = torch.from_numpy(a).long()
        if self.fix_lua_indexing:
            item -= 1  # subtract 1 for 0-based indexing
        return item

class TeacherOutputDataset(IndexedCachedDataset):
    dtype2size = {
        float: 8,
        int: 4,
    }

    def __init__(self, prefix):
        self.cache_index = {}
        super().__init__(prefix, fix_lua_indexing=False)

    @staticmethod
    def save_bin(prefix, data_list, dtype=np.float):
        bin_path = prefix + '.bin'
        idx_path = prefix + '.idx'
        builder = TeacherOutputDatasetBuilder(bin_path, dtype)
        for d in data_list:
            builder.add_item(d)
        builder.finalize(idx_path)

    def __getitem__(self, i):
        self.check_index(i)
        tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
        a = np.empty(tensor_size, dtype=self.dtype)

        if i in self.cache:
            np.copyto(a, self.cache[i])
        else:
            self.data_file.seek(self.data_offsets[i] * self.element_size)
            self.data_file.readinto(a)
            self.cache[i] = a

        item = torch.from_numpy(a)
        if self.dtype == np.int32 or self.dtype == np.int or self.dtype == np.int64:
            item = item.long()
        else:
            item = item.float()
        return item

class KdCodeSumDataset(object):

    def __init__(self, config,
                 src_dataset,expert_scores,topk_idx_dataset,topk_probs_dataset ,
                 dataset_ids,dataset_names ,is_train=False ):
        # self.opt = opt
        self.is_train = is_train
        self.src_dataset = src_dataset
        self.expert_scores = expert_scores
        self.topk_idx_dataset = topk_idx_dataset
        self.topk_probs_dataset = topk_probs_dataset
        self.dataset_ids = dataset_ids
        self.dataset_names = dataset_names
        # self.src_sizes = src_sizes
        # self.num_dataset  = len(self.src_dataset)
        self.num_dataset  = len(dataset_names)
        # LOGGER.info("naturalcode/src/data/codesum/kd_dataset.py  self.num_dataset:   ",self.num_dataset)
        self.student_scores = [0 for _ in range(self.num_dataset)]
        self.kd_threshold = config['kd']['kd_threshold']
        self.distill_topk = config['kd']['distill_topk']
        self.distill = config['kd']['distill']
        self.kd_default_alpha = config['kd']['kd_default_alpha']
        self.alpha_strategy = config['kd']['alpha_strategy']
        LOGGER.info("KdCodeSumDataset__init__  self.kd_threshold : {}".format(self.kd_threshold))

    def __getitem__(self, index):

        # sample = list(self.src_dataset[index])  # index -> id in kd_collate_fn()
        # sample.append(self.dataset_ids[index] if self.dataset_ids is not None else None)
        # sample.append(index)
        # # if self.topk_idx_dataset is not None and self.is_train:
        # if self.is_train and self.distill:
        #     assert self.topk_idx_dataset is not None
        #     sample.append(self.topk_idx_dataset[index])
        #     sample.append(self.topk_probs_dataset[index])
        #     if self.expert_scores[self.dataset_ids[index]] is None:
        #         sample.append(self.kd_default_alpha)
        #     else:
        #         # LOGGER.info("kd_dataset_getitem_student_scores: ",self.student_scores )
        #         sample.append(self.get_alpha(self.dataset_ids[index]))
        #     sample.append(self.distill_topk)
        #     sample.append('kdtrain')
        # else:
        #     sample.append('nokdtrain')

###
        new_sample = self.src_dataset[index]
        sample = []
        sample.append(self.dataset_ids[index] if self.dataset_ids is not None else None)
        sample.append(index)

        if self.is_train and self.distill:
            assert self.topk_idx_dataset is not None

            # comment_len = len(new_sample['others'][0])
            # comment_extend_vocab_len = len(new_sample['others'][1])
            # LOGGER.debug("KdCodeSumDataset_getitem index:{} self.dataset_ids[index]:{} comment_len:{} comment_extend_vocab_len:{} len(self.topk_idx_dataset[index]):{} len(self.topk_probs_dataset[index]):{} ".format(
            #     index, self.dataset_ids[index] ,    comment_len, comment_extend_vocab_len,
            #     len(self.topk_idx_dataset[index]),len(self.topk_probs_dataset[index])))


            sample.append(self.topk_idx_dataset[index])
            sample.append(self.topk_probs_dataset[index])
            if self.expert_scores[self.dataset_ids[index]] is None:
                sample.append(self.kd_default_alpha)
            else:
                # LOGGER.info("kd_dataset_getitem_student_scores: ",self.student_scores )
                sample.append(self.get_alpha(self.dataset_ids[index]))
            sample.append(self.distill_topk)
            new_sample['kdtrain'] = sample
        else:
            new_sample['nokdtrain'] = sample
####

        return new_sample

    def get_alpha(self, ds_id):
        if self.alpha_strategy == 'fix':
            return self.kd_default_alpha
        if self.alpha_strategy == 'threshold':
            # if self.student_scores[ds_id] < self.expert_scores[ds_id] + 1:
            if self.student_scores[ds_id] < self.expert_scores[ds_id] + self.kd_threshold:
                return self.kd_default_alpha
            else:
                return 0
        if self.alpha_strategy == 'adaptive':
            if self.student_scores[ds_id] < self.expert_scores[ds_id] - 2:
                return self.kd_default_alpha
            else:
                return self.kd_default_alpha * (0.5 ** (self.student_scores[ds_id] - self.expert_scores[ds_id] + 2))

    def __len__(self):
        return len(self.src_dataset)



def kd_codesum_collate_fn(batch_data: List) -> Dict:

    _, _, _, _, _, code_modalities, pointer_gen  = batch_data[0]['others']

    batch_size = len(batch_data)

# 反正naturalcodev2 rnn对于未排序的，首先排序，然后计算，然后恢复batch内样本的原来顺序，所以这里不排序，因为这里排序的话，
# 反而会引起batch 内顺序和数据集中对应的每个batch中样本顺序不一样
    # if 'tok' in code_modalities:
    #     # sort batch data, if tok available
    #     batch_data.sort(key=lambda batch: len(batch['tok'][0]), reverse=True)

    if batch_data[0].__contains__('kdtrain'):
        is_kdtrain = True
        dataset_id, id, topk_idx, topk_prob, alpha, distill_topks = \
            zip(*[batch['kdtrain'] for batch in batch_data])
    else:
        is_kdtrain = False
        dataset_id, id = \
            zip(*[batch['nokdtrain'] for batch in batch_data])

    # Build and return our designed batch (dict)
    store_batch = {'comment': [], 'pointer': [], 'case_study': []}
    for modal in code_modalities:
        store_batch[modal] = []

    # release comment first for copy-generator
    # comment
    # comment, comment_extend_vocab, raw_comment, case_study_data, parsed_code, index, _ = \
    #     zip(*[batch['others'] for batch in batch_data])


    comment, comment_extend_vocab, raw_comment, case_study_data, index, _ ,_= \
        zip(*[batch['others'] for batch in batch_data])
    comment, comment_input, comment_target, comment_len = pad_comment_sum(comment)
    if comment_extend_vocab[0] is None:  # tuple of None
        comment_extend_vocab = None
    else:
        _, _, comment_extend_vocab, _ = pad_comment_sum(comment_extend_vocab)
    # comment to tensor
    comment, comment_input, comment_target, comment_len, comment_extend_vocab, \
        = map(to_torch_long, (comment, comment_input, comment_target, comment_len, comment_extend_vocab,))
    # feed comment
    store_batch['comment'] = [comment, comment_input, comment_target, comment_len, raw_comment]

    if 'tok' in code_modalities:
        # release tok modal
        code, code_dict_comment, code_oovs = zip(*[batch['tok'] for batch in batch_data])
        code, code_len, code_mask = pad_seq(code, include_padding_mask=True)  # pad code
        code, code_len, code_mask = to_torch_long(code), to_torch_long(code_len), \
                                    torch.from_numpy(code_mask).float()
        store_batch['tok'] = [code, code_len, code_mask]

        # # for pointer
        # pointer_extra_zeros = torch.zeros(batch_size, max([len(x) for x in code_oovs]))
        # code_dict_comment, _ = pad_seq(code_dict_comment)
        # code_dict_comment = to_torch_long(code_dict_comment)
        # store_batch['pointer'] = [code_dict_comment, comment_extend_vocab, pointer_extra_zeros, code_oovs]

        if pointer_gen:
            # for pointer
            pointer_extra_zeros = torch.zeros(batch_size, max([len(x) for x in code_oovs]))
            code_dict_comment, _ = pad_seq(code_dict_comment)
            code_dict_comment = to_torch_long(code_dict_comment)
            store_batch['pointer'] = [code_dict_comment, comment_extend_vocab, pointer_extra_zeros, code_oovs]
        else:
            store_batch['pointer'] = [None, None, None, None]

    if 'ast' in code_modalities:
        # release tree/ast modal
        ast = [batch['ast'] for batch in batch_data]
        packed_graph, root_indices, node_nums, = pack_graph(ast)
        tree_mask = tree_padding_mask(node_nums)
        root_indices, node_nums, tree_mask = to_torch_long(root_indices), to_torch_long(node_nums), \
                                             to_torch_float(tree_mask)
        store_batch['ast'] = [packed_graph, root_indices, node_nums, tree_mask]

    if 'path' in code_modalities:
        # release ast-path modal
        head, center, tail = map(
            lambda batch_list: list(itertools.chain(*batch_list)),
            zip(*[batch['path'] for batch in batch_data]),
        )
        head, head_len, head_mask = pad_seq(head, include_padding_mask=True)
        head, head_len, head_mask = to_torch_long(head), to_torch_long(head_len), \
                                    to_torch_long(head_mask)

        center, center_len, center_mask = pad_seq(center, include_padding_mask=True)
        center, center_len, center_mask = to_torch_long(center), to_torch_long(center_len), \
                                          to_torch_long(center_mask)

        tail, tail_len, tail_mask = pad_seq(tail, include_padding_mask=True)
        tail, tail_len, tail_mask = to_torch_long(tail), to_torch_long(tail_len), \
                                    to_torch_long(tail_mask)

        store_batch['path'] = [
            head, head_len, head_mask,
            center, center_len, center_mask,
            tail, tail_len, tail_mask,
        ]

    # other info
    store_batch['index'] = torch.Tensor(index).long()
    # store_batch['parsed_code'] = parsed_code
    store_batch['case_study'] = case_study_data

    store_batch['dataset_id'] = dataset_id
    store_batch['id'] = id
    if is_kdtrain:
        assert comment_input.shape == comment_target.shape 
        sizes = comment_input.size(1), distill_topks[0]
        teacher_outputs = topk_idx[0][0].new(len(batch_data), *sizes).fill_(PAD).long(), \
                          topk_prob[0][0].new(len(batch_data), *sizes).fill_(PAD)

        def copy_tensor(src, dst):
            dst.copy_(src)

        assert len(topk_idx) == len(topk_prob)
        assert len(topk_idx) == len(batch_data)

        for i in range(len(topk_idx)):
            # LOGGER.info("{}/{} teacher_outputs[0].shape:{} ".format(i,len(topk_idx),teacher_outputs[0].shape ))
            vv = topk_idx[i].long()  # T * K
            # LOGGER.info("topk_idx[i].shape: {} len(vv):{}  distill_topks[0]:{} ".format(topk_idx[i].shape , len(vv),
            #                                                                             distill_topks[0] ))
            copy_tensor(vv[:, :distill_topks[0]], teacher_outputs[0][i, :len(vv), :distill_topks[0]])
            vv = topk_prob[i]
            # LOGGER.info("topk_prob[i].shape: {} len(vv):{}  distill_topks[0]:{} ".format(topk_prob[i].shape , len(vv),
            #                                                                             distill_topks[0] ))
            copy_tensor(vv[:, :distill_topks[0]], teacher_outputs[1][i, :len(vv), :distill_topks[0]])

        store_batch['teacher_output'] = teacher_outputs[0], \
                                  teacher_outputs[1]
        store_batch['alpha'] = torch.FloatTensor(alpha).reshape(-1, 1).expand(-1, comment_target.shape[1])


    return store_batch



