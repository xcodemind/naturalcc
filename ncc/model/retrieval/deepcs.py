# -*- coding: utf-8 -*-
import torch
from ncc import LOGGER
from ncc.model.template import *
from ncc.module.code2vec.encoder_tok import *
from ncc.metric import BaseLoss
from typing import Dict, Any

class DeepCodeSearch(CodeEnc_CmntEnc):
    '''
    Deep Code Search
    ref: https://guxd.github.io/papers/deepcs.pdf
    ref: https://github.com/guxd/deep-code-search
    '''

    def __init__(self, args: Dict) -> None:
        super(DeepCodeSearch, self).__init__(
            args=args,
            code_encoder=CodeEncoder_DeepCS(args),
            comment_encoder=CmntEncoder_DeepCS(args)
        )
        LOGGER.debug('building {}...'.format(self.__class__.__name__))

    def code_forward(self, batch_data: Dict, ) -> torch.Tensor:
        method, method_len, method_mask = batch_data['method']
        api, api_len, api_mask = batch_data['tok']
        code_feature = self.code_encoder(
            method, method_len, method_mask,
            api, api_len, api_mask,
        )
        return code_feature

    def comment_forward(self, batch_data: Dict, ) -> torch.Tensor:
        comment, _, _, comment_len, _ = batch_data['comment']
        comment_mask = comment.data.gt(0).to(comment.device)
        comment_feature = self.comment_encoder(comment, comment_mask)
        return comment_feature

    def train_sl(self, batch_data: Dict, criterion: BaseLoss, ) -> Any:
        code_emb = self.code_forward(batch_data)
        comment_emb = self.comment_forward(batch_data)
        loss = criterion(code_emb, comment_emb)
        return loss
