#!/usr/bin/env python3
# Copyright (c) 2019 OpenAI, HugginFace Inc. team. and TaeHwan Jung
# Copyright (c) Facebook, Inc. and its affiliates.
# ----------------------------------------------------------------------------
# MIT LICENSE
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------
"""
    Transformer model is adapted from: https://github.com/graykode/gpt-2-Pytorch
        (Commit: 46ae886391a94c6683be438269252c4afd5ba762)
    Original Paper and repository here: https://github.com/openai/gpt-2

    RNN implementation is adapted from: https://github.com/pytorch/examples/tree/master/word_language_model
"""
import torch.nn as nn
from ncc.models import register_model
from ncc.models.fairseq_model import FairseqLanguageModel
from ncc.modules.completion.transformer_encoder import TransformerEncoder


@register_model('traverse_transformer')
class TraverseTransformerModel(FairseqLanguageModel):
    def __init__(
        self,
        args,
        decoder,
        # vocab_size,
        # loss_fn,
        # n_layer,
        # n_embd,
        # n_ctx,
        # n_head,
        # layer_norm_epsilon,
        # root_paths=False,
        # rel_vocab_size=None,
    ):
        # super(TraverseTransformerModel, self).__init__()
        super().__init__(decoder)
        self.args = args
        # self.transformer = GPT2Model(
        #     vocab_size,
        #     n_layer,
        #     n_embd,
        #     n_ctx,
        #     n_head,
        #     layer_norm_epsilon,
        #     root_paths,
        #     rel_vocab_size,
        # )
        # self.lm_head = GPT2LMHead(self.transformer.wte.weight, n_embd)
        # self.loss_fn = loss_fn

    @classmethod
    def build_model(cls, args, config, task):
        """Build a new model instance."""
        # make sure all arguments are present
        # base_architecture(args)

        # if not hasattr(args, 'max_positions'):
        # if 'max_positions' not in args['model']:
        #     args['model']['max_positions'] = args['task']['tokens_per_sample']

        # encoder = RobertaEncoder(args, task.source_dictionary)
        vocab_size = 10000
        n_layer = 6
        n_embd = 768
        n_ctx = 6  # TODO
        n_head = 6
        layer_norm_epsilon = 0.01
        root_paths = None
        rel_vocab_size = None

        decoder = TransformerEncoder(
            vocab_size,
            n_layer,
            n_embd,
            n_ctx,
            n_head,
            layer_norm_epsilon,
            root_paths,
            rel_vocab_size,
            task.dictionary,
        )
        return cls(args, decoder)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, src_tokens, ext=None, rel=None, paths=None, return_loss=False
    ):
        hidden_states = self.decoder(src_tokens, rel, paths)
        # y_pred = self.lm_head(hidden_states)
        # if not return_loss:
        return hidden_states

        # ext contains a list of idx of where to take the loss from
        # we linearize it first
        # ids = []
        # max_len = y.size(-1)
        # for i, ext_i in enumerate(ext):
        #     ids += [i * max_len + j for j in range(ext_i, max_len)]
        # loss = self.loss_fn(y_pred.view(-1, y_pred.size(-1))[ids], y.view(-1)[ids])
        # return loss


class GPT2LMHead(nn.Module):
    def __init__(self, model_embeddings_weights, n_embd):
        super(GPT2LMHead, self).__init__()
        self.n_embd = n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        lm_logits = self.decoder(hidden_state)
        return lm_logits