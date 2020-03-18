# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

from src import *
from src.eval import *
from src.model.template import *
from src.module.code2vec.base import *
from src.module.code2vec.encoder_tok import *
from src.module.code2vec.multi_modal import *
from src.module.summarization import *
from src.model import *
from src.dataset import *
from src.metric import *
from src.utils.util_data import batch_to_cuda


class DeepCodeSearch(CodeEnc_CmntEnc):
    '''
    Deep Code Search
    ref: https://guxd.github.io/papers/deepcs.pdf
    ref: https://github.com/guxd/deep-code-search
    '''

    def __init__(self, config: Dict) -> None:
        super(DeepCodeSearch, self).__init__(
            config=config,
            code_encoder=CodeEncoder_DeepCS(config),
            comment_encoder=CmntEncoder_DeepCS(config)
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
