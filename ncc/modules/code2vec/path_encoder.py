import torch
import torch.nn as nn
import torch.nn.functional as F

from ncc.modules.code2vec.fairseq_encoder import FairseqEncoder
from ncc.modules.embedding import Embedding
from ncc.utils import utils

DEFAULT_MAX_SOURCE_POSITIONS = 1e5
import sys


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


class PathEncoder(FairseqEncoder):
    """
    LSTM encoder:
        head/tail -> sub_tokens -> embedding -> sum
        body -> LSTM -> hidden state
        head_sum, hidden state, tail_sum -> W -> tanh
    """

    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, num_layers=1,
        dropout_in=0.1, dropout_out=0.1, bidirectional=True,
        left_pad=True, pretrained_embed=None, padding_idx=None,
        max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS
    ):
        super().__init__(dictionary)
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.max_source_positions = max_source_positions

        self.padding_idx = padding_idx if padding_idx is not None else dictionary.pad()
        if pretrained_embed is None:
            num_embeddings = len(dictionary)
            self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad
        self.transform = nn.Linear(2 * embed_dim + 2 * hidden_size, embed_dim, bias=False)

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    @staticmethod
    def _get_sorted_order(lens):
        sorted_len, fwd_order = torch.sort(
            lens.contiguous().reshape(-1), 0, descending=True
        )
        _, bwd_order = torch.sort(fwd_order)
        return sorted_len.tolist(), fwd_order, bwd_order

    def forward(self, src_tokens, src_lengths, **kwargs):
        """head_tokens, tail_tokens, body_tokens: bsz, path_num, seqlen"""
        head_tokens, body_tokens, tail_tokens = src_tokens

        # head_tokens = head_tokens.view(-1, head_tokens.size(-1))
        # tail_tokens = tail_tokens.view(-1, tail_tokens.size(-1))

        head_repr = self.embed_tokens(head_tokens).sum(dim=-2)  # [bsz, path_num, embed_dim]
        tail_repr = self.embed_tokens(tail_tokens).sum(dim=-2)  # [bsz, path_num, embed_dim]

        # embed tokens
        x = self.embed_tokens(body_tokens)  # bsz, path_num, seqlen, embed_dim
        bsz, path_num, seqlen, embed_dim = x.size()
        x = x.view(-1, seqlen, embed_dim)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B*P x T x C -> T x B*P x C
        x = x.transpose(0, 1)
        # pack embedded source tokens into a PackedSequence
        src_lengths = src_lengths.view(-1)
        src_lengths, fwd_order, bwd_order = self._get_sorted_order(src_lengths)
        # sort seq_input & hidden state by seq_lens
        x = x.index_select(dim=1, index=fwd_order)
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths, enforce_sorted=False)

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz * path_num, self.hidden_size
        else:
            state_size = self.num_layers, bsz * path_num, self.hidden_size
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)
        _, (final_hiddens, _) = self.lstm(packed_x, (h0, c0))

        # we only use hidden_state of LSTM
        # x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_idx)
        # x = x.index_select(dim=1, index=bwd_order)
        final_hiddens = final_hiddens.index_select(dim=1, index=bwd_order)
        # final_cells = final_cells.index_select(dim=1, index=bwd_order)

        final_hiddens = final_hiddens[(-2 if self.bidirectional else -1):].transpose(dim0=0, dim1=1)
        final_hiddens = final_hiddens.contiguous().view(bsz, path_num, -1)

        x = torch.cat([final_hiddens, head_repr, tail_repr], dim=-1)
        x = torch.tanh(self.transform(x))
        final_hiddens = x.mean(dim=1).unsqueeze(dim=0)
        final_cells = torch.zeros_like(final_hiddens).to(final_hiddens.device)
        encoder_padding_mask = torch.BoolTensor(bsz, x.size(1)).to(x.device).fill_(True)

        return {
            'encoder_out': (x, final_hiddens, final_cells),
            'encoder_padding_mask': encoder_padding_mask if encoder_padding_mask.any() else None
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out['encoder_out'] = tuple(
            eo.index_select(1, new_order)
            for eo in encoder_out['encoder_out']
        )
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions
