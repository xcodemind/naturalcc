# -*- coding: utf-8 -*-


import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from ncc.modules.code2vec.ncc_encoder import NccEncoder
from ncc.modules.embedding import Embedding
from ncc.utils.pooling1d import pooling1d
from ncc.utils.activations import get_activation
from ncc.types import (
    Int_t, Sequence_t, Tensor_t,
)

logger = logging.getLogger(__name__)


class Conv1dResEncoder(NccEncoder):
    """based on CodeSearchNet """

    def __init__(self, dictionary, embed_dim: Int_t, out_channels: Sequence_t, kernel_size: Sequence_t,
                 **kwargs):
        super().__init__(dictionary)
        # word embedding + positional embedding
        self.embed = Embedding(len(dictionary), embed_dim, padding_idx=self.dictionary.pad())
        self.position_encoding = kwargs.get('position_encoding', None)
        if self.position_encoding == 'learned':
            self.position_embed = Embedding(kwargs['max_tokens'], embed_dim, padding_idx=None)
        else:
            self.position_embed = None
        # pooling
        pooling = kwargs.get('pooling', None)
        self.pooling = pooling1d(pooling)
        self.weight_layer = nn.Linear(embed_dim, 1, bias=False) if 'weighted' in pooling else None
        # conv1d
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # padding mode = ['valid'(default), 'same']
        self.padding = kwargs.get('padding', 'valid')
        if self.padding == 'same':
            self.padding_size = []
            for kernel_sz in self.kernel_size:
                padding_left = (kernel_sz - 1) // 2
                padding_right = kernel_sz - 1 - padding_left
                self.padding_size.append((padding_left, padding_right,))
        self.conv1d_layers = nn.ModuleList([])
        for idx, kernel_sz in enumerate(self.kernel_size):
            if idx == 0:
                layer = nn.Conv1d(in_channels=embed_dim, out_channels=self.out_channels[0],
                                  kernel_size=kernel_sz)
            else:
                layer = nn.Conv1d(in_channels=self.out_channels[idx - 1], out_channels=self.out_channels[idx],
                                  kernel_size=kernel_sz)
            self.conv1d_layers.append(layer)

        self.residual = kwargs.get('residual', False)  # residual
        self.dropout = kwargs.get('dropout', None)
        activation_fn = kwargs.get('activation_fn', None)
        self.activation_fn = get_activation(activation_fn) if activation_fn else None

    def forward(self, tokens: Tensor_t, tokens_len: Tensor_t = None, tokens_mask: Tensor_t = None):
        if tokens_mask is None:
            tokens_mask = (tokens.ne(self.dictionary.pad())).to(tokens.device)
        if tokens_len is None:
            tokens_len = tokens_mask.sum(dim=-1)
        tokens = self.embed(tokens)
        tokens = self._position_encoding(tokens)

        for idx, conv1d in enumerate(self.conv1d_layers):
            tokens_transposed = tokens.transpose(dim0=-2, dim1=-1)
            if self.padding == 'same':
                # TODO: check it on CodeSearchNet source code
                next_tokens = conv1d(F.pad(tokens_transposed, pad=self.padding_size[idx]))
            else:
                next_tokens = conv1d(tokens_transposed)
            next_tokens = next_tokens.transpose(dim0=-2, dim1=-1)
            # Add residual connections past the first layer.
            if self.residual and idx > 0:
                next_tokens += tokens
            if self.activation_fn:
                next_tokens = self.activation_fn(next_tokens)
            if self.dropout:
                next_tokens = F.dropout(next_tokens, p=self.dropout, training=self.training)
            tokens = next_tokens
        if self.pooling:
            tokens = self.pooling(
                input_emb=tokens, input_len=tokens_len, input_mask=tokens_mask, weight_layer=self.weight_layer
            )
        return tokens

    def _position_encoding(self, input_emb: Tensor_t) -> Tensor_t:
        if self.position_encoding == 'learned':
            pos_idx = torch.arange(start=0, end=input_emb.size(1)).unsqueeze(0).expand(input_emb.size(0), -1) \
                .to(input_emb.device)
            return input_emb + self.position_embed(pos_idx)
        elif self.position_encoding is None:
            return input_emb
        else:
            raise ValueError("Unknown position encoding '{}'!".format(self.position_encoding))
