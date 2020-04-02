# -*- coding: utf-8 -*-
import torch
from torch.nn import Module
import torch.nn.functional as F
from ncc.module.code2vec.base import Encoder_Emb, Encoder_Conv1d, pooling1d
from typing import Dict, Any

class Encoder_EmbResConv1d(Module):

    def __init__(self, token_num: int, embed_size: int,
                 out_channels: int, kernel_size: int, padding: str,
                 pooling: str, dropout=None, ) -> None:
        super(Encoder_EmbResConv1d, self).__init__()
        self.wemb = Encoder_Emb(token_num, embed_size, )

        # 2 layer
        self.conv1d1 = Encoder_Conv1d(embed_size, out_channels, kernel_size, padding, )
        self.conv1d2 = Encoder_Conv1d(out_channels, out_channels, kernel_size, padding, )

        self.pooling = pooling
        self.dropout = dropout

    @classmethod
    def load_from_config(cls, config: Dict, modal: str, ) -> Any:
        instance = cls(
            token_num=config['training']['token_num'][modal],
            embed_size=config['training']['embed_size'],
            out_channels=config['training']['conv1d_out_channels'],
            kernel_size=config['training']['conv1d_kernel_size'],
            padding=config['training']['conv1d_padding'],
            pooling=config['training']['conv1d_pooling'],
            dropout=config['training']['dropout'],
        )
        return instance

    def forward(self, input: torch.Tensor, input_mask=None) -> Any:
        if input_mask is None:
            input_mask = input.data.gt(0).float().to(input.device)
        input_emb = self.wemb(input)
        input_emb1 = F.relu(self.conv1d1(input_emb, input_mask, ))
        input_emb2 = self.conv1d2(input_emb1, input_mask, )
        input_emb = torch.tanh(input_emb1 + input_emb2)
        input_emb = F.dropout(input_emb, self.dropout, self.training)
        # pooling
        input_emb = pooling1d(input_emb, input_mask, self.pooling)
        return input_emb


if __name__ == '__main__':
    code = torch.LongTensor([[1, 2, 4, 0], [4, 3, 0, 0]])
    encoder = Encoder_EmbResConv1d(token_num=10, embed_size=50,
                                   out_channels=10, kernel_size=2, padding='same',
                                   layer_num=3, pooling='mean', act_func='tanh', dropout=0.1, )
    code_emb = encoder(code)
    print(code_emb.size())

    cmnt = torch.LongTensor([[1, 2, 4, 0], [4, 3, 0, 0]])
    decoder = Encoder_EmbResConv1d(token_num=10, embed_size=50,
                                   out_channels=10, kernel_size=2, padding='same',
                                   layer_num=3, pooling='mean', act_func='tanh', dropout=0.1, )
    cmnt_emb = decoder(cmnt)
    print(cmnt_emb.size())

    from ncc.metric.base import RetrievalNLLoss

    criterion = RetrievalNLLoss(True)

    loss = criterion(code_emb, cmnt_emb)
    loss.backward()
