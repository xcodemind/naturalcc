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

# def __init__(self,
#                  leaf_path_k: int,
#                  # for ast node, start/end ast node
#                  node_token_num: int, subtoken_token_num: int, embed_size: int,
#                  # for RNN, Linear, attention
#                  rnn_type: str, hidden_size: int, layer_num: int, dropout: float, bidirectional: bool,
#                  ) -> None:
class PathEncoder(FairseqEncoder):
    """LSTM encoder."""
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, num_layers=1,
        dropout_in=0.1, dropout_out=0.1, bidirectional=False,
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

        num_embeddings_border, num_embeddings_center = len(dictionary[0]), len(dictionary[1])
        self.padding_idx = padding_idx if padding_idx is not None else dictionary[0].pad()
        if pretrained_embed is None:
            self.embed_tokens_border = Embedding(num_embeddings_border, embed_dim, self.padding_idx)
            self.embed_tokens_center = Embedding(num_embeddings_center, embed_dim, self.padding_idx)
        else:
            self.embed_tokens_border = pretrained_embed[0]
            self.embed_tokens_center = pretrained_embed[1]

        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    # def forward(self,
    #             head: torch.Tensor, head_len: torch.Tensor, head_mask: torch.Tensor,
    #             center: torch.Tensor, center_len: torch.Tensor, center_mask: torch.Tensor,
    #             tail: torch.Tensor, tail_len: torch.Tensor, tail_mask: torch.Tensor, ) -> Any:
    # def forward(self, head_tokens, head_lengths, center_tokens, center_lengths, tail_tokens, tail_lengths):
    def forward(self, src_tokens, src_lengths):
        head_tokens, center_tokens, tail_tokens = src_tokens
        head_lengths, center_lengths, tail_lengths = src_lengths
        center_tokens = center_tokens.view(-1, center_tokens.size(-1))  # TODO
        center_lengths = center_lengths.view(-1, 1).squeeze(1)  # TODO

        if self.left_pad:
            # nn.utils.rnn.pack_padded_sequence requires right-padding;
            # convert left-padding to right-padding
            center_tokens = utils.convert_padding_direction(
                center_tokens,
                self.padding_idx,
                left_to_right=True,
            )

        bsz, seqlen = center_tokens.size()

        # embed tokens
        x = self.embed_tokens_center(center_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        print('center_lengths: ', center_lengths.size())
        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, center_lengths.data.tolist(), enforce_sorted=False) #TODO:enforce_sorted=False

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_idx)
        x = F.dropout(x, p=self.dropout_out, training=self.training)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:

            def combine_bidir(outs):
                out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
                return out.view(self.num_layers, bsz, -1)

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        encoder_padding_mask = center_tokens.eq(self.padding_idx).t()
        
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