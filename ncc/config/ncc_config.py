# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from typing import Any, Dict, List

from torch.nn.modules.loss import _Loss

# from src import metrics
from ncc.logging import metrics
from ncc.utils import utils


class NccConfig(object):

    def __init__(self):
        # super().__init__()
        # self.task = task
        # if hasattr(task, 'target_dictionary'):
        #     tgt_dict = task.target_dictionary
        #     self.padding_idx = tgt_dict.pad() if tgt_dict is not None else -100
        super().__init__()
    # @staticmethod
    # def add_args(parser):
    #     """Add criterion-specific arguments to the parser."""
    #     pass

    @classmethod
    def build_config(cls, config, task):
        raise NotImplementedError("Model must implement the build_model method")
