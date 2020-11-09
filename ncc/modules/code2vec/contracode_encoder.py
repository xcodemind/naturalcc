from typing import Optional, Tuple
import math

import torch
from torch import nn
from ncc.modules.code2vec.ncc_encoder import NccEncoder
from ncc.modules.code2vec.lstm_encoder import LSTMEncoder
import torch.nn.functional as F
from ncc.utils import utils
from ncc.modules.attention.multihead_attention import MultiheadAttention
from ncc.modules.code2vec.transformer_encoder_layer import TransformerEncoderLayer
from ncc.modules.roberta.positional_embedding_bak import PositionalEmbedding
from ncc.modules.roberta.layer_norm import LayerNorm
import random
DEFAULT_MAX_SOURCE_POSITIONS = 1e5
from ncc.modules.code2vec.ncc_encoder import EncoderOut


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.v_proj.weight.data.normal_(mean=0.0, std=0.02)


class CodeEncoderTransformer(NccEncoder):
    def __init__(
            self,
            args,
            dictionary,
            embed_tokens,
            num_segments: int = 2,
            offset_positions_by_padding: bool = False, # True,
            apply_bert_init: bool = False,
            freeze_embeddings: bool = False,
            n_trans_layers_to_freeze: int = 0,
            export: bool = False,
            traceable: bool = False,
    ):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self.args = args
        self.dropout = args['model']['dropout']
        self.encoder_layerdrop = args['model']['encoder_layerdrop']

        self.embed_dim = embed_tokens.embedding_dim
        self.padding_idx = dictionary.pad() # embed_tokens.padding_idx TODO
        # self.vocab_size = vocab_size
        self.max_source_positions = args['model']['max_source_positions']

        self.embed_tokens = embed_tokens
        self.embed_scale = 1.0 if args['model']['no_scale_embedding'] else math.sqrt(self.embed_dim)

        # Option 1
        if args['model']['position_encoding_version'] == 'ncc_sinusoidal':
            from ncc.modules.roberta.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
            self.embed_positions = SinusoidalPositionalEmbedding(
                self.embed_dim,
                padding_idx= self.padding_idx, #(self.padding_idx if offset_positions_by_padding else None),
                init_size=args['model']['max_source_positions'] + self.padding_idx + 1 if offset_positions_by_padding else args['model']['max_source_positions'],  #  + 1 why?
            )
        # Option 2
        elif args['model']['position_encoding_version'] == 'ncc_learned':
            from ncc.modules.roberta.learned_positional_embedding import LearnedPositionalEmbedding
            if self.padding_idx is not None:
                num_embeddings = args['model']['max_source_positions'] + self.padding_idx + 1
            m = LearnedPositionalEmbedding(num_embeddings, self.embed_dim, self.padding_idx)
            nn.init.normal_(m.weight, mean=0, std=self.embed_dim ** -0.5)
            if self.padding_idx is not None:
                nn.init.constant_(m.weight[self.padding_idx], 0)
            self.embed_positions = m
        # Option 3
        elif args['model']['position_encoding_version'] == 'contracode':
            from ncc.modules.roberta.sinusoidal_positional_embedding_simple import SinusoidalPositionalEmbedding_Simple
            self.embed_positions = SinusoidalPositionalEmbedding_Simple(self.embed_dim, args['model']['dropout'], max_len=args['model']['max_source_positions'])

        self.num_segments = num_segments
        self.segment_embeddings = (
            nn.Embedding(self.num_segments, self.embed_dim, padding_idx=None)
            if self.num_segments > 0
            else None
        )
        torch.manual_seed(1)
        # Option 1 (The NCC Option 2 has been the same as Option 1 in logistic)
        # encoder_layer = nn.TransformerEncoderLayer(self.embed_dim, args['model']['encoder_attention_heads'],
        #                                            args['model']['encoder_ffn_embed_dim'], 0, args['model']['activation_fn']) # args['model']['dropout']
        encoder_layer = TransformerEncoderLayer(args)
        self.layers = nn.ModuleList([encoder_layer for i in range(args['model']['encoder_layers'])])
        # Option 2
        # self.layers = nn.ModuleList([TransformerEncoderLayer(args) for i in range(args['model']['encoder_layers'])])

        self.num_layers = len(self.layers)
        # if args['model']['encoder_normalize_before']:
        self.layer_norm = nn.LayerNorm(self.embed_dim)  # LayerNorm(self.embed_dim) TODO
        # else:
        #     self.layer_norm = None
        if args['model']['layernorm_embedding']:  # getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = nn.LayerNorm(self.embed_dim)     # LayerNorm(self.embed_dim, export=export) TODO
        else:
            self.layernorm_embedding = None

        self.traceable = traceable

    def forward_embedding(self, src_tokens, positions, segment_labels, padding_mask):
        x = self.embed_tokens(src_tokens)

        if self.embed_scale is not None:
            x = embed = x * self.embed_scale

        if self.embed_positions is not None:
            # x += self.embed_positions(src_tokens, positions=positions)
            # x += self.embed_positions(src_tokens)  # TODO, position里面如果已经+=了，这里就没必要了
            if self.args['model']['position_encoding_version'] == 'contracode':
                x += self.embed_positions(x)
            elif self.args['model']['position_encoding_version'] == 'ncc_sinusoidal':
                # x += self.embed_positions(src_tokens, positions=positions)
                pe = self.embed_positions(src_tokens, positions=positions)
                x += pe
        if self.segment_embeddings is not None and segment_labels is not None:
            x += self.segment_embeddings(segment_labels)

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        # x = F.dropout(x, p=self.dropout, training=self.training) #TODO, position里面如果已经dropout了，这里就没必要了

        # # account for padding while computing the representation
        # if padding_mask is not None:
        #     x *= 1 - padding_mask.unsqueeze(-1).type_as(x)

        return x, embed

    def forward_(self, src_tokens, lengths=None, no_project_override=False):
        # output = self.encoder(src_tokens, lengths, no_project_override)
        # return output
        src_emb = self.embedding(src_tokens).transpose(0, 1) * math.sqrt(self.config["d_model"])
        src_emb = self.embed_positions(src_emb)
        if self.pad_id is not None:
            src_key_padding_mask = src_tokens == self.pad_id #.config["pad_id"]
        else:
            src_key_padding_mask = None
        out = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)  # TxBxD
        if not no_project_override and self.project:  #.config["project"]:
            return self.project_layer(out.mean(dim=0))
        else:
            return out

    def forward(
            self,
            src_tokens: torch.Tensor,
            segment_labels: torch.Tensor = None,
            last_state_only: bool = False,
            positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # compute padding mask. This is needed for multi-head attention
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not self.traceable and not encoder_padding_mask.any():
            encoder_padding_mask = None

        x, encoder_embedding = self.forward_embedding(src_tokens, positions, segment_labels, encoder_padding_mask)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []
        # if not last_state_only:
        #     encoder_states.append(x)

        for layer in self.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            # dropout_probability = random.uniform(0, 1)
            # if not self.training or (dropout_probability > self.encoder_layerdrop):
            x = layer(x, encoder_padding_mask) # TODO
            # x = layer(x, src_mask=None, src_key_padding_mask=encoder_padding_mask)
            if not last_state_only:
                encoder_states.append(x)

        # if self.layer_norm is not None:
        x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
        )


        # sentence_rep = x[0, :, :]  # <CLS> presentation
        #
        # if last_state_only:
        #     encoder_states = [x]
        #
        # if self.traceable:
        #     return torch.stack(encoder_states), sentence_rep
        # else:
        #     return encoder_states, sentence_rep


class CodeEncoderLSTM(NccEncoder):
    pass
