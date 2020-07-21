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
from ncc.types import (
    Int_t, Tensor_t,
)

logger = logging.getLogger(__name__)


class NBOWEncoder(FairseqEncoder):
    """based on CodeSearchNet """

    def __init__(self, dictionary, embed_dim: Int_t, **kwargs):
        super().__init__(dictionary)
        self.embed = Embedding(len(dictionary), embed_dim, padding_idx=self.dictionary.pad())
        # self.dropout = kwargs.get('dropout', None)
        pooling = kwargs.get('pooling', None)
        self.pooling = pooling1d(pooling)
        self.weight_layer = Linear(embed_dim, 1, bias=False) if 'weighted' in pooling else None

    def forward(self, tokens: Tensor_t, tokens_len: Tensor_t = None, tokens_mask: Tensor_t = None):
        """
        Args:
            tokens: [batch_size, max_token_len]
            tokens_mask: [batch_size, 1]
            tokens_mask: [batch_size, max_token_len]

        Returns:
            tokens: [batch_size, max_token_len, embed_dim]
        """
        if tokens_mask is None:
            tokens_mask = (tokens.ne(self.dictionary.pad())).to(tokens.device)
        if tokens_len is None:
            tokens_len = tokens_mask.sum(dim=-1)
        tokens = self.embed(tokens)
        if self.pooling:
            tokens = self.pooling(
                input_emb=tokens, input_len=tokens_len, input_mask=tokens_mask, weight_layer=self.weight_layer
            )
        return tokens
