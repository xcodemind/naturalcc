# -*- coding: utf-8 -*-

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from ncc.data.tools import data_utils
from ncc.data.dictionary import Dictionary
from ncc.tasks import register_task
from ncc.tasks.fairseq_task import FairseqTask
from ncc.utils.tokenizer import tokenize_string
from ncc.data.retrieval.retrieval_dataset import RetrievalDataset

# from ncc.types import *
from ncc.types import Dict_t, Class_t, String_t, Int_t, Bool_t
from ncc.utils import utils
from ncc.utils import path


def load_tokens_dataset(data_path, split, src, src_dict, tgt, tgt_dict, dataset_impl,
                        src_max_tokens=None, tgt_max_tokens=None):
    src_path = os.path.join(data_path, '{}.{}'.format(split, src))
    src_dataset = data_utils.load_indexed_dataset(src_path, 'tok', src_dict, dataset_impl)
    tgt_path = os.path.join(data_path, '{}.{}'.format(split, tgt))
    tgt_dataset = data_utils.load_indexed_dataset(tgt_path, 'tok', tgt_dict, dataset_impl)
    return RetrievalDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset.sizes, tgt_dict,
        max_source_positions=src_max_tokens, max_target_positions=tgt_max_tokens,
    )


@register_task('retrieval')
class RetrievalTask(FairseqTask):
    """
    Task for code retrieval models (e.g., code and docstring).
    A simple implementation. Only consider (single modality) code-comment retrieval task.
    """

    def __init__(self, args: Dict_t, src_dict, tgt_dict):
        super().__init__(args)
        self.args = args
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.seed = args['common']['seed']

    @classmethod
    def setup_task(cls, args: Dict_t, **kwargs) -> Class_t:
        paths = utils.split_paths(args['task']['data'])
        assert len(paths) > 0
        if args['dataset']['joined_dictionary']:
            modalities = sorted(args['task']['source_lang'] + args['task']['target_lang'])
            src_dict = tgt_dict = cls.load_dictionary(path.join(paths[0], '{}.dict.json'.format('_'.join(modalities))))
        else:
            src_dict = cls.load_dictionary(path.join(paths[0], '{}.dict.json'.format(args['task']['source_lang'])))
            tgt_dict = cls.load_dictionary(path.join(paths[0], '{}.dict.json'.format(args['task']['target_lang'])))
        return cls(args, src_dict, tgt_dict)

    @classmethod
    def build_dictionary(
        cls, filenames, tokenize_func=tokenize_string,
        workers=1, threshold=-1, nwords=-1, padding_factor=8
    ):
        """Build the dictionary

        Args:
            filenames (list): list of filenames
            workers (int): number of concurrent workers
            threshold (int): defines the minimum word count
            nwords (int): defines the total number of words in the final dictionary,
                including special symbols
            padding_factor (int): can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        d = Dictionary()

        for filename in filenames:
            Dictionary.add_token_to_dictionary(
                filename, d, tokenize_func, workers
            )

        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

    def load_dataset(self, split: String_t, epoch: Int_t = 1, combine: Bool_t = False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args['task']['data'])
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        src, tgt = self.args['task']['source_lang'], self.args['task']['target_lang']

        if self.args['model']['arch'] in ['nbow', 'conv1d_res', 'rnn']:
            self.datasets[split] = load_tokens_dataset(
                data_path, split, src, self.source_dictionary, tgt, self.target_dictionary,
                dataset_impl=self.args['dataset']['dataset_impl'],
                src_max_tokens=self.args['dataset']['code_max_tokens'],
                tgt_max_tokens=self.args['dataset']['query_max_tokens'],
            )
        else:
            raise NotImplementedError

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
