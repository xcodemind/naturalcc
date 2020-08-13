# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""
import sys
import torch
import torch.nn as nn
from ncc.models import register_model
from ncc.models.contracode.contracode_moco import ContraCodeMoCo


@register_model('contracode_hybrid')
class ContraCodeHybrid(ContraCodeMoCo):

    @classmethod
    def hub_models(cls):
        return {
            'roberta.base': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz',
            'roberta.large': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz',
            'roberta.large.mnli': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.mnli.tar.gz',
            'roberta.large.wsc': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.wsc.tar.gz',
        }

    def __init__(self, args, q_encoder, k_encoder):
        super().__init__(q_encoder, k_encoder)
        self.args = args

        # # We follow BERT's random weight initialization
        # self.apply(init_bert_params)

        # self.classification_heads = nn.ModuleDict()
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.register_buffer("queue", torch.randn(d_rep, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    @classmethod
    def build_model(cls, args, config, task):
        """Build a new model instance."""

        # make sure all arguments are present
        # base_architecture(args)

        # if not hasattr(args, 'max_positions'):
        if 'max_positions' not in args['model']:
            args['model']['max_positions'] = args['task']['tokens_per_sample']

        encoder_params = {}
        q_encoder = cls.make_encoder(**encoder_params)
        k_encoder = cls.make_encoder(**encoder_params)
        return cls(args, q_encoder, k_encoder)

    def mlm_forward(self, im):  # predicted masked tokens
        features = self.encoder_q(im, no_project_override=True)  # L x B x D
        assert len(features.shape) == 3, str(features.shape)
        L, B, D = features.shape
        assert D == self.d_model
        features = self.mlm_head(features).view(L, B, D)  # L x B x D
        logits = torch.matmul(features, self.encoder_q.embedding.weight.transpose(0, 1)).view(L, B,
                                                                                              self.n_tokens)  # [L, B, ntok]
        return torch.transpose(logits, 0, 1).view(B, L, self.n_tokens)  # [B, T, ntok]

    def moco_forward(self, im_q, im_k):  # logits, labels
        return super().forward(im_q, im_k)

    def forward(self, im_q, im_k):
        predicted_masked_tokens = self.mlm_forward(im_q)
        moco_logits, moco_targets = self.moco_forward(im_q, im_k)
        return predicted_masked_tokens, moco_logits, moco_targets

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args['model']['max_positions']
