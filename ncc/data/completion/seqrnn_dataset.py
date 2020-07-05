# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from ncc.data.tools import data_utils
from ncc.data.fairseq_dataset import FairseqDataset


def collate(samples, pad_idx, eos_idx):
    # no need for left padding
    if len(samples) == 0:
        return {}

    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx,
        )

    src_tokens = merge('source')
    tgt_tokens = merge('target')
    node_ids = {name: [] for name in samples[0]['node_id'].keys()}

    max_len = max(len(sample['source']) for sample in samples)
    max_len = max(max_len, 2)
    # TODO: illustrate this code, pls. do not understand
    for i, sample in enumerate(samples):
        for name, lst in sample['node_id'].items():
            node_ids[name] += [j - 1 + (max_len - 1) * i for j in lst]

    # because split a AST into sub-ASTs
    # loss mask: 1) skip padding idx and 2) begin counting at start idx
    loss_mask = (src_tokens != pad_idx).to(src_tokens.device)  # 1) skip padding idx
    # 2) begin counting at start idx
    for idx in range(len(samples)):
        loss_mask[idx, :samples[idx]['start_idx']] = False
    loss_mask = loss_mask.view(-1)

    ntokens = loss_mask.sum().item()

    batch = {
        'net_input': {
            'src_tokens': src_tokens,
            # 'src_lengths': src_lengths,
        },
        'target': tgt_tokens,
        'node_ids': node_ids,
        'ntokens': ntokens,
        'loss_mask': loss_mask,
        'id': [s['id'] for s in samples],
        # 'nsentences': 2,
        # 'sample_size': masked_ids.size(0),
    }
    return batch


class SeqRNNDataset(FairseqDataset):
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
            self, src, src_sizes, src_dict, node_ids,
            tgt=None, tgt_sizes=None, tgt_dict=None,
            left_pad_source=False, left_pad_target=False,
            # max_source_positions=1024, max_target_positions=1024,
            shuffle=True, input_feeding=True,
            remove_eos_from_source=False, append_eos_to_target=False,
            # align_dataset=None,
            append_bos=False, eos=None,
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
        self.node_ids = node_ids
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        # self.max_source_positions = max_source_positions
        # self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        # self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        # self.align_dataset = align_dataset
        # if self.align_dataset is not None:
        #     assert self.tgt_sizes is not None, "Both source and target needed when alignments are provided"
        self.append_bos = append_bos
        self.eos = (eos if eos is not None else src_dict.eos())

    def __getitem__(self, index):
        # Append EOS to end of tgt sentence if it does not have an EOS
        # and remove EOS from end of src sentence if it exists.
        # This is useful when we use existing datasets for opposite directions
        #   i.e., when we want to use tgt_dataset as src_dataset and vice versa
        src_item = self.src[index][:-1]
        tgt_item = self.src[index][1:]

        # if self.append_eos_to_target:
        #     eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
        #     if self.tgt and self.tgt[index][-1] != eos:
        #         tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])
        #
        # if self.append_bos:
        #     # for src/tgt
        #     bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
        #     if self.tgt and self.tgt[index][0] != bos:
        #         tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])
        #
        #     bos = self.src_dict.bos()
        #     if self.src[index][-1] != bos:
        #         src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])
        #
        # if self.remove_eos_from_source:
        #     eos = self.src_dict.eos()
        #     if self.src[index][-1] == eos:
        #         src_item = self.src[index][:-1]

        node_id = self.node_ids[index]
        start_idx = self.src.start_idx[index]
        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,
            'node_id': node_id,
            'start_idx': start_idx,
        }
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
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.eos,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        # return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)
        return self.src_sizes[index]  # + self.tgt_sizes[index]

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
