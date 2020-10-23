# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F

from ncc.types import *
from ncc.models import register_model
from ncc.models.ncc_model import NccRetrievalModel
from ncc.modules.retrieval.rnn_encoder import RNNEncoder

import logging

logger = logging.getLogger(__name__)


@register_model('birnn')
class RNN(NccRetrievalModel):
    def __init__(self, args, src_encoder, tgt_encoder):
        super().__init__(src_encoder, tgt_encoder)
        self.args = args

    @classmethod
    def build_model(cls, args, config, task):
        """dictionary, embed_dim: Int_t, out_channels: Sequence_t, kernel_size: Sequence_t,"""
        src_encoder = RNNEncoder(
            dictionary=task.source_dictionary, embed_dim=args['model']['code_embed_dim'],
            rnn_cell=args['model']['code_rnn_cell'], rnn_hidden_dim=args['model']['code_rnn_hidden_dim'],
            rnn_num_layers=args['model']['code_rnn_num_layers'],
            rnn_bidirectional=args['model']['code_rnn_bidirectional'],

            max_tokens=args['dataset']['code_max_tokens'], dropout=args['model']['dropout'],
            pooling=args['model']['code_pooling'], rnn_dropout=args['model']['code_rnn_dropout'],
        )
        tgt_encoder = RNNEncoder(
            dictionary=task.target_dictionary, embed_dim=args['model']['query_embed_dim'],
            rnn_cell=args['model']['query_rnn_cell'], rnn_hidden_dim=args['model']['query_rnn_hidden_dim'],
            rnn_num_layers=args['model']['query_rnn_num_layers'],
            rnn_bidirectional=args['model']['query_rnn_bidirectional'],

            max_tokens=args['dataset']['query_max_tokens'], dropout=args['model']['dropout'],
            pooling=args['model']['query_pooling'], rnn_dropout=args['model']['query_rnn_dropout'],
        )
        return cls(args, src_encoder, tgt_encoder)
