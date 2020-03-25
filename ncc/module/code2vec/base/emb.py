# -*- coding: utf-8 -*-

import sys

sys.path.append('./')

from ncc import *
from ncc.module.code2vec.base.util import *

from ncc.utils.constants import PAD


class Encoder_Emb(Module):
    '''
    neural bag of words
    for code/comment embedding
    ref: DeepCodeSeach and CodeResearchNet
    '''

    def __init__(self, token_num: int, embed_size: int, ) -> None:
        super(Encoder_Emb, self).__init__()
        self.embedding = nn.Embedding(token_num, embed_size, padding_idx=PAD)

    @classmethod
    def load_from_config(cls, config: Dict, modal: str, ) -> Any:
        instance = cls(
            token_num=config['training']['token_num'][modal],
            embed_size=config['training']['embed_size'],
        )
        return instance

    def forward(self, input: torch.Tensor, input_mask=None, ):
        input_emb = self.embedding(input)  # input: [batch_sz x seq_len x 1]  embedded: [batch_sz x seq_len x emb_sz]
        return input_emb


if __name__ == '__main__':
    input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    print(input.size())

    emb_encoder = Encoder_Emb(10, 3, pooling='max')
    input = emb_encoder(input)
    print(input.size())
