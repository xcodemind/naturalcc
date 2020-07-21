# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F

from ncc.types import *
from ncc.models import register_model
from ncc.models.fairseq_model import FairseqRetrievalModel
from ncc.modules.retrieval.conv1d_res_encoder import Conv1dResEncoder

import logging

logger = logging.getLogger(__name__)


@register_model('conv1d_res')
class Conv1dRes(FairseqRetrievalModel):
    def __init__(self, args, src_encoder, tgt_encoder):
        super().__init__(src_encoder, tgt_encoder)
        self.args = args

    @classmethod
    def build_model(cls, args, config, task):
        """dictionary, embed_dim: Int_t, out_channels: Sequence_t, kernel_size: Sequence_t,"""
        src_encoder = Conv1dResEncoder(
            dictionary=task.source_dictionary, embed_dim=args['model']['code_embed_dim'],
            out_channels=args['model']['code_layers'], kernel_size=args['model']['code_kernel_size'],

            max_tokens=args['dataset']['code_max_tokens'],
            dropout=args['model']['dropout'], residual=args['model']['code_residual'],
            activation_fn=args['model']['code_activation_fn'], padding=args['model']['code_paddding'],
            pooling=args['model']['code_pooling'], position_encoding=args['model']['code_position_encoding'],
        )
        tgt_encoder = Conv1dResEncoder(
            dictionary=task.target_dictionary, embed_dim=args['model']['query_embed_dim'],
            out_channels=args['model']['query_layers'], kernel_size=args['model']['query_kernel_size'],

            max_tokens=args['dataset']['query_max_tokens'],
            dropout=args['model']['dropout'], residual=args['model']['query_residual'],
            activation_fn=args['model']['query_activation_fn'], padding=args['model']['query_paddding'],
            pooling=args['model']['query_pooling'], position_encoding=args['model']['query_position_encoding'],
        )
        return cls(args, src_encoder, tgt_encoder)
