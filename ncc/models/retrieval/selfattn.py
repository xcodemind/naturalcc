# -*- coding: utf-8 -*-
import torch
from ncc.models.template import *
from ncc.module.code2vec.encoder_tok import *
from ncc.metric import *
from typing import Dict, Any


class RNNSelfAttn(CodeEnc_CmntEnc):

    def __init__(self, args: Dict):
        super(RNNSelfAttn, self).__init__(
            args=args,
            code_encoder=Encoder_EmbRNNSelfAttn.load_from_config(args, modal='tok'),
            comment_encoder=Encoder_EmbRNNSelfAttn.load_from_config(args, modal='tok'),
        )

    def code_forward(self, batch_data: Dict, ) -> Any:
        code, code_len, _ = batch_data['tok']
        output = self.code_encoder(code, input_len=code_len, )
        return output

    def comment_forward(self, batch_data: Dict, ) -> Any:
        comment, _, _, comment_len, _ = batch_data['comment']
        output = self.comment_encoder(comment, input_len=comment_len, )
        return output

    def train_sl(self, batch_data: Dict, criterion: BaseLoss, ) -> torch.Tensor:
        code_emb = self.code_forward(batch_data)
        comment_emb = self.comment_forward(batch_data)
        loss = criterion(code_emb, comment_emb)
        return loss


if __name__ == '__main__':
    pass
