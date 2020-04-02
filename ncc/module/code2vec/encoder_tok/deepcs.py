# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import Module
from ncc.module.code2vec.base import pooling1d
from ncc.module.code2vec.encoder_tok import Encoder_EmbRNN
from typing import Dict, Any


class CodeEncoder_DeepCS(Module):
    def __init__(self, config: Dict, ):
        super(CodeEncoder_DeepCS, self).__init__()
        self.method_encoder = Encoder_EmbRNN.load_from_config(config, modal='method')
        self.api_encoder = Encoder_EmbRNN.load_from_config(config, modal='tok')
        self.pooling = 'max'
        self.fusion = nn.Linear(
            2 * config['training']['rnn_hidden_size'] * (2 if config['training']['rnn_bidirectional'] else 1),
            config['training']['rnn_hidden_size'] * (2 if config['training']['rnn_bidirectional'] else 1),
        )

    def forward(self, method: torch.Tensor, method_len: torch.Tensor, method_mask: torch.Tensor,
                api: torch.Tensor, api_len: torch.Tensor, api_mask: torch.Tensor, ) -> Any:
        method_feature, _ = self.method_encoder(method, method_len, )
        method_feature = torch.tanh(method_feature)
        method_feature = pooling1d(method_feature, method_mask, self.pooling)

        api_feature, _ = self.api_encoder(api, api_len, )
        api_feature = torch.tanh(api_feature)
        api_feature = pooling1d(api_feature, api_mask, self.pooling)

        fused_feature = torch.cat([method_feature, api_feature], dim=-1)
        code_feature = self.fusion(fused_feature)
        return code_feature


class CmntEncoder_DeepCS(Module):
    def __init__(self, config: Dict, ):
        super(CmntEncoder_DeepCS, self).__init__()
        self.desc_encoder = Encoder_EmbRNN.load_from_config(config, modal='comment')
        self.pooling = 'max'

    def forward(self, desc: torch.Tensor, desc_mask: torch.Tensor, ) -> Any:
        desc_feature, _ = self.desc_encoder(desc)
        desc_feature = torch.tanh(desc_feature)
        desc_feature = pooling1d(desc_feature, desc_mask, self.pooling)
        return desc_feature
