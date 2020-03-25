# -*- coding: utf-8 -*-

import sys

sys.path.append('./')

from ncc import *
from ncc.model.template import *
from ncc.metric import *
from ncc.module.code2vec.encoder_tok import Encoder_EmbResConv1d


class ResConv1d(CodeEnc_CmntEnc):

    def __init__(self, config: Dict):
        super(ResConv1d, self).__init__(
            config=config,
            code_encoder=Encoder_EmbResConv1d.load_from_config(config, modal='tok'),
            comment_encoder=Encoder_EmbResConv1d.load_from_config(config, modal='comment'),
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
