# -*- coding: utf-8 -*-
import torch
from ncc.model.template import *
from ncc.metric import *
from ncc.module.code2vec.encoder_tok import Encoder_EmbResConv1d
from typing import Dict, Any

class ResConv1d(CodeEnc_CmntEnc):

    def __init__(self, args: Dict):
        super(ResConv1d, self).__init__(
            args=args,
            code_encoder=Encoder_EmbResConv1d.load_from_config(args, modal='tok'),
            comment_encoder=Encoder_EmbResConv1d.load_from_config(args, modal='comment'),
        )

    def code_forward(self, batch_data: Dict, ) -> Any:
        code, _, code_mask = batch_data['tok']
        return self.code_encoder(code, code_mask)

    def comment_forward(self, batch_data: Dict, ) -> Any:
        comment = batch_data['comment'][0]
        return self.comment_encoder(comment)

    def train_sl(self, batch_data: Dict, criterion: BaseLoss, ) -> torch.Tensor:
        code_emb = self.code_forward(batch_data)
        comment_emb = self.comment_forward(batch_data)
        loss = criterion(code_emb, comment_emb)
        return loss


if __name__ == '__main__':
    pass
