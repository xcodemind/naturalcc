# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ncc.modules.roberta.layer_norm import LayerNorm
from ncc.modules.attention.multihead_attention import MultiheadAttention
from ncc.modules.roberta.positional_embedding import PositionalEmbedding
from ncc.modules.roberta.transformer_sentence_encoder_layer import TransformerSentenceEncoderLayer
from ncc.modules.code2vec.transformer_encoder_layer import TransformerEncoderLayer
import random
from ncc.modules.code2vec.fairseq_encoder import FairseqEncoder
import math

DEFAULT_MAX_SOURCE_POSITIONS = 1e5


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


class TransformerSentenceEncoder(FairseqEncoder):
    """
    Implementation for a Bi-directional Transformer based Sentence Encoder used
    in BERT/XLM style pre-trained models.

    This first computes the token embedding using the token embedding matrix,
    position embeddings (if specified) and segment embeddings
    (if specified). After applying the specified number of
    TransformerEncoderLayers, it outputs all the internal states of the
    encoder as well as the final representation associated with the first
    token (usually CLS token).

    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens

    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape T x B x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        num_segments: int = 2,
        offset_positions_by_padding: bool = True,
        apply_bert_init: bool = False,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
        traceable: bool = False,
    ):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout = args['model']['dropout']
        self.encoder_layerdrop = args['model']['encoder_layerdrop']

        self.embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        # self.vocab_size = vocab_size
        self.max_source_positions = args['model']['max_source_positions']

        self.embed_tokens = embed_tokens
        self.embed_scale = 1.0 if args['model']['no_scale_embedding'] else math.sqrt(self.embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args['model']['max_source_positions'],
                self.embed_dim,
                padding_idx=(self.padding_idx if offset_positions_by_padding else None),
                learned=args['model']['encoder_learned_pos'],
            )
            if not args['model']['no_token_positional_embeddings']
            else None
        )

        self.num_segments = num_segments
        self.segment_embeddings = (
            nn.Embedding(self.num_segments, self.embed_dim, padding_idx=None)
            if self.num_segments > 0
            else None
        )
        # self.embed_tokens = nn.Embedding(
        #     self.vocab_size, self.embedding_dim, self.padding_idx
        # )
        # self.layers = nn.ModuleList(
        #     [
        #         TransformerSentenceEncoderLayer(
        #             embedding_dim=self.embedding_dim,
        #             ffn_embedding_dim=ffn_embedding_dim,
        #             num_attention_heads=num_attention_heads,
        #             dropout=self.dropout,
        #             attention_dropout=attention_dropout,
        #             activation_dropout=activation_dropout,
        #             activation_fn=activation_fn,
        #             add_bias_kv=add_bias_kv,
        #             add_zero_attn=add_zero_attn,
        #             export=export,
        #         )
        #         for _ in range(num_encoder_layers)
        #     ]
        # )
        self.layers = nn.ModuleList([TransformerEncoderLayer(args) for i in range(args['model']['encoder_layers'])])
        self.num_layers = len(self.layers)

        if args['model']['layernorm_embedding']:
            self.layernorm_embedding = LayerNorm(self.embed_dim, export=export)
        else:
            self.layernorm_embedding = None

        self.apply_bert_init = apply_bert_init
        # self.learned_pos_embedding = learned_pos_embedding
        self.traceable = traceable
        # Apply initialization of model params after building the model
        if self.apply_bert_init:
            self.apply(init_bert_params)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        if freeze_embeddings:
            freeze_module_params(self.embed_tokens)
            freeze_module_params(self.segment_embeddings)
            freeze_module_params(self.embed_positions)
            freeze_module_params(self.layernorm_embedding)

        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

    def forward_embedding(self, src_tokens, positions, segment_labels, padding_mask):
        x = self.embed_tokens(src_tokens)

        if self.embed_scale is not None:
            x *= self.embed_scale

        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens, positions=positions)

        if self.segment_embeddings is not None and segment_labels is not None:
            x += self.segment_embeddings(segment_labels)

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # # account for padding while computing the representation
        # if padding_mask is not None:
        #     x *= 1 - padding_mask.unsqueeze(-1).type_as(x)

        return x

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

        x = self.forward_embedding(src_tokens, positions, segment_labels, encoder_padding_mask)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []
        # if not last_state_only:
        #     encoder_states.append(x)

        for layer in self.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not self.training or (dropout_probability > self.encoder_layerdrop):
                x = layer(x, encoder_padding_mask)
                if not last_state_only:
                    encoder_states.append(x)

        sentence_rep = x[0, :, :]  # <CLS> presentation

        if last_state_only:
            encoder_states = [x]

        if self.traceable:
            return torch.stack(encoder_states), sentence_rep
        else:
            return encoder_states, sentence_rep
