import math

import torch
from torch import nn
from ncc.modules.code2vec.fairseq_encoder import FairseqEncoder

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


class CodeEncoder(nn.Module):
    def __init__(
        self,
        n_tokens,
        d_model=512,
        d_rep=256,
        n_head=8,
        n_encoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        activation="relu",
        norm=True,
        pad_id=None,
        project=False,
    ):
        super().__init__()
        self.config = {k: v for k, v in locals().items() if k != "self"}
        self.embedding = nn.Embedding(n_tokens, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=9000)
        norm_fn = nn.LayerNorm(d_model) if norm else None
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout, activation)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers, norm=norm_fn)
        if project:
            self.project_layer = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_rep))
        # NOTE: We use the default PyTorch intialization, so no need to reset parameters.

    def forward(self, x, lengths=None, no_project_override=False):
        src_emb = self.embedding(x).transpose(0, 1) * math.sqrt(self.config["d_model"])
        src_emb = self.pos_encoder(src_emb)
        if self.config["pad_id"] is not None:
            src_key_padding_mask = x == self.config["pad_id"]
        else:
            src_key_padding_mask = None
        out = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)  # TxBxD
        if not no_project_override and self.config["project"]:
            return self.project_layer(out.mean(dim=0))
        else:
            return out


class CodeEncoderLSTM(FairseqEncoder):
    def __init__(
        self,
        args,
        dictionary,
        # Deeptyper set d_model to 200
        embed_dim=512,
        hidden_size=512,
        # d_rep=512,
        num_layers=2,
        dropout=0.1,
        padding_idx=None,
        project='hidden',
        max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS

    ):
        super().__init__(dictionary)
        self.max_source_positions = max_source_positions
        num_embeddings = len(dictionary)
        self.padding_idx = padding_idx if padding_idx is not None else dictionary.pad()


        self.config = {k: v for k, v in locals().items() if k != "self"}
        self.embed_tokens = nn.Embedding(num_embeddings, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_len=9000)

        # Currently using 2 layers of LSTM
        print(f"CodeEncoderLSTM: Creating BiLSTM with {num_layers} layers, {embed_dim} hidden and input size")
        # TODO: Apply dropout to LSTM
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)

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
        # B, T = src_tokens.size(0), src_tokens.size(1)
        src_emb = self.embed_tokens(src_tokens).transpose(0, 1) * math.sqrt(self.config["hidden_size"])
        src_emb = self.pos_encoder(src_emb)

        # Compute sequence lengths and pack src_emb
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(src_emb, src_lengths, enforce_sorted=False)

        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x)  # TxBxD
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_outs)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        # if x.size(0) != T:
        #     print("WARNING: unexpected size of encoder output: out.shape=", x.shape, "x.shape=", src_tokens.shape,
        #           "src_emb.shape=", src_emb.shape)
        #     print("lengths.max()=", src_lengths.max())
        #     print("lengths.min()=", src_lengths.min())


        if not no_project_override and self.config["project"]:
            if self.config["project"] == "sequence_mean":
                # out is T x B x n_directions*d_model
                rep = x.mean(dim=0)  # B x n_directions*d_model
            elif self.config["project"] == "sequence_mean_nonpad":
                out_ = x.transpose(0, 1)  # B x T x n_directions*d_model
                mask = torch.arange(out_.size(1), device=out_.device).unsqueeze(0).unsqueeze(-1).expand_as(
                    out_) < src_lengths.unsqueeze(1).unsqueeze(2)
                rep = (out_ * mask.float()).sum(dim=1)  # B x n_directions*d_model
                rep = rep / src_lengths.unsqueeze(1).float()
            elif self.config["project"] == "hidden":
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