# -*- coding: utf-8 -*-

import sys

sys.path.append('./')

from ncc import *
from ncc.model.template import *
from ncc.module.code2vec.base import Encoder_Emb
from ncc.metric import *
from ncc.module.code2vec.base.util import *


class NBOW(CodeEnc_CmntEnc):

    def __init__(self, config: Dict):
        super(NBOW, self).__init__(
            config=config,
            code_encoder=Encoder_Emb.load_from_config(config, modal='tok'),
            comment_encoder=Encoder_Emb.load_from_config(config, modal='comment'),
        )

    def code_forward(self, batch_data: Dict, ) -> Any:
        code, _, code_mask = batch_data['tok']
        code_emb = torch.tanh(self.code_encoder(code))
        code_emb = pooling1d(code_emb, code_mask, 'max')
        # code_emb = torch.tanh(code_emb)
        return code_emb

    def comment_forward(self, batch_data: Dict, ) -> Any:
        comment = batch_data['comment'][0]
        comment_emb = torch.tanh(self.comment_encoder(comment))
        comment_mask = comment.data.gt(0)
        comment_emb = pooling1d(comment_emb, comment_mask, 'max')
        # comment_emb = torch.tanh(comment_emb)
        return comment_emb

    def train_sl(self, batch_data: Dict, criterion: BaseLoss, ) -> torch.Tensor:
        code_emb = self.code_forward(batch_data)
        comment_emb = self.comment_forward(batch_data)
        loss = criterion(code_emb, comment_emb)
        return loss


if __name__ == '__main__':
    nn.LSTM
