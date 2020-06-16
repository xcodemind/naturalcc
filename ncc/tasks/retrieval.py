# -*- coding: utf-8 -*-

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from ncc.tasks import register_task
from ncc.tasks.fairseq_task import FairseqTask

from ncc.types import *

from ncc.utils import utils
from ncc.utils import path


def load_dataset(args: Dict_t, ):
    ...


@register_task('')
class RetrievalTask(FairseqTask):
    """Task for code retrieval models (e.g., code and docstring)."""

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
        # dictionary = Dictionary.load(os.path.join(paths[0], 'dict.code.txt'))
        if args['dataset']['joined_dictionary']:
            tgt_dict = src_dict = cls.load_dictionary(
                path.join(paths[0], '{}.dict.txt'.format(args['task']['source_lang']))
            )
        else:
            src_dict = cls.load_dictionary(path.join(paths[0], '{}.dict.txt'.format(args['task']['source_lang'])))
            tgt_dict = cls.load_dictionary(path.join(paths[0], '{}.dict.txt'.format(args['task']['target_lang'])))
        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split: String_t, epoch: Int_t = 1, combine: Bool_t = False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args['task']['data'])
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        src, tgt = self.args['task']['source_lang'], self.args['task']['target_lang']
