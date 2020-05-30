# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import OrderedDict
import numpy as np
import torch
from ncc.data.tools import data_utils
from ncc.data.fairseq_dataset import FairseqDataset


logger = logging.getLogger(__name__)

SENT_SEP = '<S_SEP>'


def find_sep(src, sep_id):
    sep_pos = []
    for i, v in enumerate(src):
        if sep_id == v:
            sep_pos.append(i)
    return sep_pos


# tokens are left-padding
def docs2tensor(docs, pad_idx):
    doc_sep_pos = map(lambda x: x[1], docs)
    max_nsent = max(map(len, doc_sep_pos))
    srcs = map(lambda x: x[0], docs)
    max_seqlen = max(map(len, srcs))
    bsz = len(docs)
    # print('max_nsent', max_nsent)
    # print('max_seqlen', max_seqlen)
    src_tokens = torch.LongTensor(bsz, max_seqlen).fill_(pad_idx)
    doc_pad_mask = torch.ByteTensor(bsz, max_nsent).fill_(1)
    src_sent_ends = torch.LongTensor(bsz, max_nsent).fill_(0)   # assume default sentence ends (for padding) are 0s
    for i in range(bsz):
        src, sep_pos = docs[i]
        src_tokens[i, 0:len(src)] = src
        doc_pad_mask[i, 0:len(sep_pos)] = 0
        src_sent_ends[i, 0:len(sep_pos)] = torch.LongTensor(sep_pos)

    return src_tokens, doc_pad_mask, src_sent_ends


def create_src_tok_batch(samples, sep_id, eos_idx, pad_idx):
    docs = []
    for sample in samples:
        src = sample['source']
        if src[-1] != sep_id:
            src_len = src.size(0)
            new_src = src.new(src_len + 1)
            new_src[0:src_len] = src
            new_src[-1] = sep_id
            src = new_src

        sep_pos = find_sep(src, sep_id)
        docs.append((src, sep_pos))

    return docs2tensor(docs, pad_idx)


def create_target_batch(samples, pad_idx):
    maxlen = max([len(s['target']) for s in samples])
    bsz = len(samples)
    target = torch.LongTensor(bsz, maxlen).fill_(pad_idx)
    for i, s in enumerate(samples):
        tgt = s['target']
        tgt_len = len(tgt)
        target[i, 0:tgt_len] = tgt
    return target


def collate(samples, src_dict, tgt_dict, left_pad_source=True, left_pad_target=False):
    if len(samples) == 0:
        return {}

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens, doc_pad_mask, src_sent_ends = create_src_tok_batch(samples, src_dict.index(SENT_SEP), src_dict.eos(), src_dict.pad())

    # print('src_tokens', src_tokens.size())
    # print('doc_pad_mask', doc_pad_mask.size())
    # print( src_tokens[:, :, -1] )
    doc_pos_tok = torch.LongTensor(doc_pad_mask.size() ).fill_( src_dict.index(SENT_SEP))
    doc_pos_tok[doc_pad_mask] = src_dict.pad()
    # print( '** doc_pos_tok **' )
    # print( doc_pos_tok )

    ntokens = sum(len(s['target']) for s in samples)
    target = create_target_batch(samples, tgt_dict.pad())

    prev_output_tokens = target  # TODO
    # print('target', target.size())

    return {
        'id': id,
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_sent_ends': src_sent_ends,
            'doc_pad_mask': doc_pad_mask,
            'doc_pos_tok': doc_pos_tok,
            'prev_output_tokens': prev_output_tokens,
        },
        'target': target,
    }


class HiCodePairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
    """

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True,
        remove_eos_from_source=False, append_eos_to_target=False,
        align_dataset=None,
        append_bos=False, eos=None
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert self.tgt_sizes is not None, "Both source and target needed when alignments are provided"
        self.append_bos = append_bos
        self.eos = (eos if eos is not None else src_dict.eos())

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][-1] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,
        }
        if self.align_dataset is not None:
            example['alignment'] = self.align_dataset[index]
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        # return collate(
        #     samples, pad_idx=self.src_dict.pad(), eos_idx=self.eos,
        #     left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
        #     input_feeding=self.input_feeding,
        # )
        return collate(
            samples, src_dict=self.src_dict, tgt_dict=self.tgt_dict,
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            # input_feeding=self.input_feeding,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)
