# -*- coding: utf-8 -*-
import torch
from ncc.models.template import *
from ncc.module.code2vec.encoder_tok import *
from ncc.module.code2vec.multi_modal import *
from ncc.metric import BaseLoss
from typing import Dict, Any


class MMAN(CodeEnc_CmntEnc):
    '''
    multi-modalities attention network
    ref: Multi-Modal Attention Network Learning for Semantic Source Code Retrieval
    '''

    def __init__(self, args: Dict, ):
        super(MMAN, self).__init__(
            args=args,
            code_encoder=CodeEnocder_MM(args),
            # code_encoder=MMEncoder_EmbRNN(args),
            # RNN -> tanh
            comment_encoder=Encoder_EmbRNN.load_from_config(args, modal='comment'),
        )

    def code_forward(self, batch_data: Dict, ) -> torch.Tensor:
        # enc_output, dec_hc, enc_mask = self.code_encoder(batch_data)
        # return self.code_encoder(batch_data)
        return self.code_encoder(batch_data)

    def comment_forward(self, batch_data: Dict, ) -> torch.Tensor:
        comment, _, _, comment_len, _ = batch_data['comment']
        comment_emb, (hidden, _) = self.comment_encoder(comment, comment_len)
        comment_emb, _ = comment_emb.max(dim=1)
        return comment_emb
        # comment_emb = torch.tanh(hidden.view(comment.size(0), -1))
        # return comment_emb

    def train_sl(self, batch_data: Dict, criterion: BaseLoss, ) -> Any:
        code_emb = self.code_forward(batch_data)
        comment_emb = self.comment_forward(batch_data)
        loss = criterion(code_emb, comment_emb)
        return loss
