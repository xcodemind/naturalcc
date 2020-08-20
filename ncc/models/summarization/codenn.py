import torch
import torch.nn as nn
import torch.nn.functional as F

from ncc.models.fairseq_model import FairseqEncoderDecoderModel
from ncc.modules.code2vec.fairseq_encoder import FairseqEncoder
from ncc.modules.seq2seq.fairseq_incremental_decoder import FairseqIncrementalDecoder
from ncc.modules.embedding import Embedding
from ncc.models import register_model
from ncc.utils import utils

DEFAULT_MAX_SOURCE_POSITIONS = 1e5
DEFAULT_MAX_TARGET_POSITIONS = 1e5


class NBOWEncoder(FairseqEncoder):
    def __init__(
        self, dictionary, embed_dim=512, dropout=0.1,
        pretrained_embed=None, padding_idx=None,
        max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS,
    ):
        super().__init__(dictionary)
        self.dropout = dropout
        self.max_source_positions = max_source_positions

        num_embeddings = len(dictionary)
        self.padding_idx = padding_idx if padding_idx is not None else dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed

    def forward(self, src_tokens, src_lengths, **kwargs):
        # embed tokens
        x = self.embed_tokens(src_tokens)
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        return {
            'encoder_out': (x,),
            'encoder_padding_mask': encoder_padding_mask if encoder_padding_mask.any() else None
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out['encoder_out'] = tuple(
            eo.index_select(0, new_order)
            for eo in encoder_out['encoder_out']
        )
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions


def Linear(in_features, out_features, bias=True, dropout=0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


class LSTMDecoder(FairseqIncrementalDecoder):
    """LSTM decoder."""

    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1,
        encoder_output_units=512, pretrained_embed=None,
        share_input_output_embed=False,
        max_target_positions=DEFAULT_MAX_TARGET_POSITIONS
    ):
        super().__init__(dictionary)
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.max_target_positions = max_target_positions

        num_embeddings = len(dictionary)
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx=dictionary.pad())
        else:
            self.embed_tokens = pretrained_embed

        # disable input feeding if there is no encoder
        # input feeding is described in arxiv.org/abs/1508.04025
        self.layers = nn.ModuleList([
            LSTMCell(
                input_size=encoder_output_units if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])

        # W_H(h)+W_T(t) => fc_out
        self.W_H = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_T = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        if not self.share_input_output_embed:
            self.fc_out = Linear(out_embed_dim, num_embeddings)

    def attention(self, hidden, encoder_out, encoder_mask):
        """
        hidden: [batch_size, hidden_size]
        encoder_out: [batch_size, src_len, hidden_size]
        encoder_mask: [batch_size, src_len]
        """
        # influence of src_hidden(j) over tgt_hidden_i
        a_ij = torch.bmm(encoder_out, hidden.unsqueeze(dim=-1)).squeeze(dim=-1)  # [batch_size, src_len]
        if encoder_mask is None:
            a_ij_softmax = a_ij.softmax(dim=-1)
        else:
            a_ij_softmax = a_ij.masked_fill(encoder_mask, float('-inf')).softmax(dim=-1)
        a_ij_softmax = a_ij_softmax.unsqueeze(dim=1)  # [batch_size, 1, src_len]
        # [batch_size, 1, src_len] x [batch_size, src_len, hidden_size] => [batch_size, 1, hidden_size]
        t_i = torch.bmm(a_ij_softmax, encoder_out).squeeze(dim=1)
        return t_i

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        x, attn_scores = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state
        )
        return self.output_layer(x), attn_scores

    def extract_features(
        self, prev_output_tokens, encoder_out, incremental_state=None
    ):
        """
        Similar to *forward* but only return features.
        """
        encoder_padding_mask = encoder_out.get('encoder_padding_mask', None)
        encoder_out = encoder_out.get('encoder_out', None)

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, tgt_len = prev_output_tokens.size()

        # get outputs from encoder
        encoder_outs = encoder_out[0]

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # setup zero cells, since there is no encoder
        num_layers = len(self.layers)
        prev_hiddens = [x.new_zeros(bsz, self.hidden_size) for i in range(num_layers)]
        prev_cells = [x.new_zeros(bsz, self.hidden_size) for i in range(num_layers)]

        outs = []
        for j in range(tgt_len):
            input = x[j]

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            attn_out = self.attention(hidden, encoder_outs, encoder_padding_mask)
            attn_out = F.dropout(attn_out, p=self.dropout_out, training=self.training)

            out = torch.tanh(self.W_H(hidden) + self.W_T(attn_out))

            # save final output
            outs.append(out)

        # collect outputs across time steps
        x = torch.stack(outs, dim=0)
        # T x B x C -> B x T x C
        x = x.transpose(1, 0)
        return x, None

    def output_layer(self, x):
        """Project features to the vocabulary size."""
        x = self.fc_out(x)
        return x

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions


@register_model('codenn')
class CodeNNModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder: NBOWEncoder, decoder: LSTMDecoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, args, config, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        # base_architecture(args)

        max_source_positions = args['model']['max_source_positions'] \
            if args['model']['max_source_positions'] else DEFAULT_MAX_SOURCE_POSITIONS
        max_target_positions = args['model']['max_target_positions'] \
            if args['model']['max_target_positions'] else DEFAULT_MAX_TARGET_POSITIONS

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        if args['model']['encoder_embed']:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args['model']['encoder_embed_path'], task.source_dictionary, args['model']['encoder_embed_dim'])
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_encoder_embed = Embedding(
                num_embeddings, args['model']['encoder_embed_dim'], task.source_dictionary.pad()
            )

        if args['model']['share_all_embeddings']:
            # double check all parameters combinations are valid
            if task.source_dictionary != task.target_dictionary:
                raise ValueError('--share-all-embeddings requires a joint dictionary')
            if args['model']['decoder_embed_path'] and (
                args['model']['decoder_embed_path'] != args['model']['encoder_embed_path']):
                raise ValueError(
                    '--share-all-embed not compatible with --decoder-embed-path'
                )
            if args['model']['encoder_embed_dim'] != args['model']['decoder_embed_dim']:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to '
                    'match --decoder-embed-dim'
                )
            pretrained_decoder_embed = pretrained_encoder_embed
            args['model']['share_decoder_input_output_embed'] = True
        else:
            # separate decoder input embeddings
            pretrained_decoder_embed = None
            if args['model']['decoder_embed']:
                pretrained_decoder_embed = load_pretrained_embedding_from_file(
                    args['model']['decoder_embed'],
                    task.target_dictionary,
                    args['model']['decoder_embed_dim']
                )
        # one last double check of parameter combinations
        if args['model']['share_decoder_input_output_embed'] and (
            args['model']['decoder_embed_dim'] != args['model']['decoder_out_embed_dim']):
            raise ValueError(
                '--share-decoder-input-output-embeddings requires '
                '--decoder-embed-dim to match --decoder-out-embed-dim'
            )

        if args['model']['encoder_freeze_embed']:
            pretrained_encoder_embed.weight.requires_grad = False
        if args['model']['decoder_freeze_embed']:
            pretrained_decoder_embed.weight.requires_grad = False

        encoder = NBOWEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args['model']['encoder_embed_dim'],
            dropout=args['model']['encoder_dropout'],
            pretrained_embed=pretrained_encoder_embed,
            max_source_positions=max_source_positions
        )
        decoder = LSTMDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args['model']['decoder_embed_dim'],
            hidden_size=args['model']['decoder_hidden_size'],
            out_embed_dim=args['model']['decoder_out_embed_dim'],
            num_layers=args['model']['decoder_layers'],
            dropout_in=args['model']['decoder_dropout_in'],
            dropout_out=args['model']['decoder_dropout_out'],
            encoder_output_units=args['model']['encoder_embed_dim'],
            pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=args['model']['share_decoder_input_output_embed'],
            max_target_positions=max_target_positions
        )
        return cls(encoder, decoder)
