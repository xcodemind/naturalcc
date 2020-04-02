# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from ncc.module.code2vec.base import Encoder_Emb
from typing import Dict, Any

class Encoder_EmbMaxpool(nn.Module):
    '''
    for code/comment embedding
    ref: DeepCodeSeach and CodeResearchNet
    '''

    def __init__(self, token_num: int, embed_size: int, dropout: float, ) -> None:
        super(Encoder_EmbMaxpool, self).__init__()
        self.dropout = dropout
        self.wemb = Encoder_Emb(token_num, embed_size)

    @classmethod
    def load_from_config(cls, config: Dict) -> Any:
        instance = cls(
            token_num=config['training']['code_token_num'],
            embed_size=config['training']['embed_size'],
            dropout=config['training']['dropout'],
        )
        return instance

    def forward(self, input: torch.Tensor):
        seq_len = input.size(-1)
        input_emd = self.wemb(input)  # input: [batch_sz x seq_len x 1]  embedded: [batch_sz x seq_len x emb_sz]
        input_emd = F.dropout(input_emd, self.dropout, training=self.training)  # [batch_size x seq_len x emb_size]
        input_emd = F.max_pool1d(input_emd.transpose(1, 2), seq_len).squeeze(2)  # [batch_size x emb_size]
        input_emd = torch.tanh(input_emd)
        return input_emd


if __name__ == '__main__':
    input = torch.LongTensor([[1, 2, 4, 0], [4, 3, 0, 0]])
    encoder = Encoder_EmbMaxpool(token_num=10, embed_size=20, dropout=0.1)
    output = encoder(input)
    print(output.size())
