# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from ncc.data.tools import data_utils
from ncc.data.fairseq_dataset import FairseqDataset


def collate(samples, pad_idx):
    # no need for left padding
    if len(samples) == 0:
        return {}

    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
        )

    src_tokens = merge('source')
    tgt_tokens = merge('target')
    node_ids = {name: [] for name in samples[0]['node_id'].keys()}
    extends = []
    max_len = max(len(sample['source']) for sample in samples)
    max_len = max(max_len, 2)

    for i, sample in enumerate(samples):
        extends.append(sample['extend'])
        for name, lst in sample['node_id'].items():
            node_ids[name] += [j - 1 + (max_len - 1) * i for j in lst]

    ntokens = sum(len(s['target']) for s in samples)

    batch = {
        'net_input': {
            'src_tokens': src_tokens,
        },
        'target': tgt_tokens,
        'node_ids': node_ids,
        "extends": extends,
        'ntokens': ntokens,
        'id': [s['id'] for s in samples],
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
            self, tgt, tgt_sizes, tgt_dict, node_ids, extends,
            left_pad_source=False, left_pad_target=False,
            max_target_positions=1024,
            shuffle=True,
    ):
        self.tgt = tgt
        self.tgt_sizes = np.array(tgt_sizes)
        self.tgt_dict = tgt_dict
        self.node_ids = node_ids
        self.extends = extends
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle

    def __getitem__(self, index):
        # Append EOS to end of tgt sentence if it does not have an EOS
        # and remove EOS from end of src sentence if it exists.
        # This is useful when we use existing datasets for opposite directions
        #   i.e., when we want to use tgt_dataset as src_dataset and vice versa
        src_item = self.tgt[index][:-1]
        tgt_item = self.tgt[index][1:]

        node_id = self.node_ids[index]
        extend = self.extends[index]
        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,
            'node_id': node_id,
            'extend': extend,
        }
        return example

    def __len__(self):
        return len(self.tgt)

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
        return collate(
            samples, pad_idx=self.tgt_dict.pad()
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.tgt_sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.tgt_sizes[index]
