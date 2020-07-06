# -*- coding: utf-8 -*-

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
from torch.nn import Linear
import torch.nn.functional as F
from ncc.models import register_model
from ncc.modules.code2vec.fairseq_encoder import FairseqEncoder
from ncc.modules.embedding import Embedding
from ncc.utils.pooling1d import pooling1d
from ncc.types import *

logger = logging.getLogger(__name__)


class NBOWEncoder(FairseqEncoder):
    """based on CodeSearchNet """

    def __init__(self, dictionary, embed_dim: Int_t, dropout: Float_t,
                 pooling: String_t = None):
        super().__init__(dictionary)
        self.embed = Embedding(len(dictionary), embed_dim, padding_idx=self.dictionary.pad())
        self.dropout = dropout
        self.pooling = pooling1d(pooling)
        self.weight_layer = Linear(embed_dim, 1) if 'weighted' in pooling else None

    def forward(self, tokens: Tensor_t, tokens_len: Tensor_t = None, tokens_mask: Tensor_t = None):
        """

        Args:
            tokens: [batch_size, max_token_len]
            tokens_mask: [batch_size, 1]
            tokens_mask: [batch_size, max_token_len]

        Returns:

        """
        if tokens_mask is None:
            tokens_mask = (tokens.ne(self.dictionary.pad())).to(tokens.device)
        if tokens_len is None:
            tokens_len = tokens_mask.sum(dim=-1)
        # type transform for dropout
        tokens = self.embed(tokens)
        # tokens = F.dropout(tokens, p=self.dropout, training=self.training)
        if self.pooling:
            tokens = self.pooling(
                input_emb=tokens, input_len=tokens_len, input_mask=tokens_mask, weight_layer=self.weight_layer
            )
            # input_weights = (self.weight_layer(tokens)).sigmoid()
            # # input_weights = torch.sigmoid(kwargs['weight_layer'](input_emb))  # B x T x 1
            # input_weights *= tokens_mask.unsqueeze(dim=-1)  # B x T x 1
            # input_emb_weighted_sum = (tokens * input_weights).sum(dim=1)  # B x D
            # return input_emb_weighted_sum / input_weights.sum(dim=1).clamp(1e-8)  # B x D
        return tokens
