import math

import torch
from torch import nn
from ncc.modules.code2vec.fairseq_encoder import FairseqEncoder
from ncc.modules.code2vec.lstm_encoder import LSTMEncoder
import torch.nn.functional as F
from ncc.utils import utils

DEFAULT_MAX_SOURCE_POSITIONS = 1e5


class PositionalEncoding(nn.Module):
    """From https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""

    def __init__(self, d_model, dropout=0.1, max_len=9000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)

    def _load_from_state_dict(self, *args):
        print("PositionalEncoding: doing nothing on call to _load_from_state_dict")


# class CodeEncoder(nn.Module):
#     def __init__(
#         self,
#         n_tokens,
#         d_model=512,
#         d_rep=256,
#         n_head=8,
#         n_encoder_layers=6,
#         d_ff=2048,
#         dropout=0.1,
#         activation="relu",
#         norm=True,
#         pad_id=None,
#         project=False,
#     ):
#         super().__init__()
#         self.config = {k: v for k, v in locals().items() if k != "self"}
#         self.embedding = nn.Embedding(n_tokens, d_model)
#         self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=9000)
#         norm_fn = nn.LayerNorm(d_model) if norm else None
#         encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout, activation)
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers, norm=norm_fn)
#         if project:
#             self.project_layer = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_rep))
#         # NOTE: We use the default PyTorch intialization, so no need to reset parameters.
#
#     def forward(self, x, lengths=None, no_project_override=False):
#         src_emb = self.embedding(x).transpose(0, 1) * math.sqrt(self.config["d_model"])
#         src_emb = self.pos_encoder(src_emb)
#         if self.config["pad_id"] is not None:
#             src_key_padding_mask = x == self.config["pad_id"]
#         else:
#             src_key_padding_mask = None
#         out = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)  # TxBxD
#         if not no_project_override and self.config["project"]:
#             return self.project_layer(out.mean(dim=0))
#         else:
#             return out


class CodeEncoderTransformer(FairseqEncoder):
    def __init__(self,
                 # args,
                 source_dictionary,
                 d_model=512,
                 d_rep=128,
                 n_head=8,
                 n_encoder_layers=6,
                 d_ff=2048,
                 dropout=0.1,
                 activation="relu",
                 norm=True,
                 # pad_id=None,
                 # encoder_type="transformer"
                 project=False,
                 ):
        super().__init__(source_dictionary)
        self.n_tokens = len(source_dictionary)
        self.pad_id = source_dictionary.pad()
        self.project = project
        # self.encoder = CodeEncoder(len(source_dictionary), d_model, d_rep, n_head, n_encoder_layers, d_ff, dropout,
        #                            activation, norm, padding_idx, project=project)
        self.config = {k: v for k, v in locals().items() if k != "self"}
        torch.manual_seed(1)
        self.embedding = nn.Embedding(self.n_tokens, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=9000)
        norm_fn = nn.LayerNorm(d_model) if norm else None
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout, activation)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers, norm=norm_fn)
        if project:
            self.project_layer = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_rep))

    def forward(self, src_tokens, lengths, no_project_override=False):
        # output = self.encoder(src_tokens, lengths, no_project_override)
        # return output
        src_emb = self.embedding(src_tokens).transpose(0, 1) * math.sqrt(self.config["d_model"])
        src_emb = self.pos_encoder(src_emb)
        if self.pad_id is not None:
            src_key_padding_mask = src_tokens == self.pad_id #.config["pad_id"]
        else:
            src_key_padding_mask = None
        out = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)  # TxBxD
        if not no_project_override and self.project:  #.config["project"]:
            return self.project_layer(out.mean(dim=0))
        else:
            return out


class CodeEncoderLSTM(LSTMEncoder):
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, num_layers=1,
        dropout_in=0.1, dropout_out=0.1, bidirectional=False,
        left_pad=True, pretrained_embed=None, padding_idx=None,
        max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS, project='hidden',
    ):
        super().__init__(dictionary, embed_dim, hidden_size, num_layers,
        dropout_in, dropout_out, bidirectional,
        left_pad, pretrained_embed, padding_idx,
        max_source_positions)

        self.project = project
        if project:
            if project == "sequence_mean" or project == "sequence_mean_nonpad":
                project_in = 2 * hidden_size
                self.project_layer = nn.Sequential(nn.Linear(project_in, hidden_size), nn.ReLU(), nn.Linear(embed_dim, 128)) # 218->hidden_size
            elif project == "hidden":
                project_in = num_layers * 2 * hidden_size
                self.project_layer = nn.Sequential(nn.Linear(project_in, hidden_size), nn.ReLU(), nn.Linear(embed_dim, 128))
            # elif project == "hidden_identity":
            #     pass
            else:
                raise ValueError(f"Unknown value '{project}' for CodeEncoderLSTM project argument")
        # NOTE: We use the default PyTorch intialization, so no need to reset parameters.

    def forward(self, src_tokens, src_lengths, no_project_override=False):
        self.lstm.flatten_parameters()
        if self.left_pad:
            # nn.utils.rnn.pack_padded_sequence requires right-padding;
            # convert left-padding to right-padding
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                self.padding_idx,
                left_to_right=True,
            )

        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)
        # x = self.pos_encoder(x) # TODO

        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist(), enforce_sorted=False)

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
        # assert list(x.size()) == [seqlen, bsz, self.output_units]  # TODO

        if self.bidirectional:
            def combine_bidir(outs):
                out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
                return out.view(self.num_layers, bsz, -1)

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        if not no_project_override and self.project:
            if self.project == "sequence_mean":
                # out is T x B x n_directions*d_model
                rep = x.mean(dim=0)  # B x n_directions*d_model
            elif self.project == "sequence_mean_nonpad":
                out_ = x.transpose(0, 1)  # B x T x n_directions*d_model
                mask = torch.arange(out_.size(1), device=out_.device).unsqueeze(0).unsqueeze(-1).expand_as(
                    out_) < src_lengths.unsqueeze(1).unsqueeze(2)
                rep = (out_ * mask.float()).sum(dim=1)  # B x n_directions*d_model
                rep = rep / src_lengths.unsqueeze(1).float()
            elif self.project == "hidden":
                # h_n is n_layers*n_directions x B x d_model
                rep = torch.flatten(final_hiddens.transpose(0, 1), start_dim=1)
            # elif self.config["project"] == "hidden_identity"
            #     return torch.flatten(h_n.transpose(0, 1), start_dim=1)
            else:
                raise ValueError
            # return self.project_layer(rep)
            return {
                'encoder_out': (self.project_layer(rep), final_hiddens, final_cells),
                'encoder_padding_mask': encoder_padding_mask if encoder_padding_mask.any() else None
            }

        # return out
        return {
            'encoder_out': (x, final_hiddens, final_cells),
            'encoder_padding_mask': encoder_padding_mask if encoder_padding_mask.any() else None
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions