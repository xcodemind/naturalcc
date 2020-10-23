# -*- coding: utf-8 -*-


import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ncc.modules.code2vec.ncc_encoder import NccEncoder
from ncc.modules.embedding import Embedding
from ncc.utils.pooling1d import pooling1d
from ncc.utils.activations import get_activation
from ncc.types import (
    Int_t, Sequence_t, Tensor_t, String_t, Bool_t, Any_t,
)

logger = logging.getLogger(__name__)


class RNNEncoder(NccEncoder):
    """based on CodeSearchNet """

    def __init__(self, dictionary, embed_dim: Int_t,
                 # rnn config
                 rnn_cell: String_t, rnn_hidden_dim: Int_t, rnn_num_layers: Int_t = 1,
                 rnn_bidirectional: Bool_t = False,
                 **kwargs):
        super().__init__(dictionary)
        # word embedding + positional embedding
        self.embed = Embedding(len(dictionary), embed_dim, padding_idx=self.dictionary.pad())
        # pooling
        pooling = kwargs.get('pooling', None)
        self.pooling = pooling1d(pooling)
        self.weight_layer = nn.Linear(embed_dim, 1, bias=False) if 'weighted' in pooling else None
        self.dropout = kwargs.get('dropout', None)
        self.rnn = getattr(nn, rnn_cell.upper())(
            embed_dim, rnn_hidden_dim, num_layers=rnn_num_layers,
            dropout=kwargs.get('rnn_dropout', None),  # rnn inner dropout between layers
            bidirectional=rnn_bidirectional,
        )

    def init_hidden(self, batch_size: int, ) -> Any_t:
        weight = next(self.parameters()).data
        biRNN = 2 if self.rnn.bidirectional else 1
        if isinstance(self.rnn, nn.LSTM):
            return (
                weight.new(self.layer_num * biRNN, batch_size, self.hidden_size).zero_().requires_grad_(),
                weight.new(self.layer_num * biRNN, batch_size, self.hidden_size).zero_().requires_grad_()
            )
        else:
            return weight.new(self.layer_num * biRNN, batch_size, self.hidden_size).zero_().requires_grad_()

    def _get_sorted_order(self, lens: Tensor_t) -> Any_t:
        sorted_len, fwd_order = torch.sort(
            lens.contiguous().reshape(-1), 0, descending=True
        )
        _, bwd_order = torch.sort(fwd_order)
        sorted_len = list(sorted_len)
        return sorted_len, fwd_order, bwd_order

    def _dynamic_forward(self, input_emb: Tensor_t, input_len: Tensor_t, hidden_state=None) -> Any_t:
        # 1) unsorted seq_emb, 2) padding to same length
        sorted_len, fwd_order, bwd_order = self._get_sorted_order(input_len)
        # sort seq_input & hidden state by seq_lens
        sorted_input_emb = input_emb.index_select(dim=0, index=fwd_order)
        if hidden_state is None:
            pass
        else:
            if isinstance(self.rnn, nn.LSTM):
                hidden_state = (
                    hidden_state[0].index_select(dim=1, index=fwd_order),
                    hidden_state[1].index_select(dim=1, index=fwd_order),
                )
            else:
                hidden_state = hidden_state.index_select(dim=1, index=fwd_order)

        sorted_input_emb = sorted_input_emb.transpose(dim0=0, dim1=1)
        packed_seq_input = pack_padded_sequence(sorted_input_emb, sorted_len)

        packed_seq_output, hidden_state = self.rnn(packed_seq_input, hidden_state)

        unpacked_seq_output, _ = pad_packed_sequence(packed_seq_output)
        unpacked_seq_output = unpacked_seq_output.index_select(dim=1, index=bwd_order)
        unpacked_seq_output = unpacked_seq_output.transpose(dim0=0, dim1=1)

        # restore seq_input & hidden state by seq_lens
        if isinstance(self.rnn, nn.LSTM):
            hidden_state = (
                hidden_state[0].index_select(dim=1, index=bwd_order),
                hidden_state[1].index_select(dim=1, index=bwd_order),
            )
        else:
            hidden_state = hidden_state.index_select(dim=1, index=bwd_order)
        return unpacked_seq_output, hidden_state

    def _merge_state(self, hidden_state) -> Any_t:
        if isinstance(self.rnn, nn.LSTM):
            hidden_state = torch.cat(
                [hc.contiguous().view(hc.size(0), -1) for hc in hidden_state],
                dim=-1
            )
        else:
            hidden_state = torch.cat(
                [torch.cat(*hs, dim=-1) for hs in hidden_state],
                dim=-1
            )
        return hidden_state

    def forward(self, tokens: Tensor_t, tokens_len: Tensor_t = None, tokens_mask: Tensor_t = None) -> Tensor_t:
        # TODO: need to be checked
        if tokens_mask is None:
            tokens_mask = (tokens.ne(self.dictionary.pad())).to(tokens.device)
        if tokens_len is None:
            tokens_len = tokens_mask.sum(dim=-1)
        tokens = self.embed(tokens)
        if self.dropout:
            tokens = F.dropout(tokens, p=self.dropout, training=self.training)
        tokens, _ = self._dynamic_forward(tokens, tokens_len)
        if self.pooling:
            tokens = self.pooling(
                input_emb=tokens, input_len=tokens_len, input_mask=tokens_mask, weight_layer=self.weight_layer
            )
        return tokens
