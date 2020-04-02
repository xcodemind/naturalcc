# -*- coding: utf-8 -*-
import abc
from torch.nn import Module
from ncc.model.template.model_template import IModel
from typing import Dict, Any, Union, List

class CodeEnc_CmntEnc(IModel):
    '''
    code retrieval
    double encoder for code/comment
    '''

    def __init__(self, config: Dict, code_encoder: Union[Module, List], comment_encoder: Union[Module, List], ):
        self.config = config
        super(CodeEnc_CmntEnc, self).__init__()
        self.code_encoder = code_encoder
        self.comment_encoder = comment_encoder

    @abc.abstractmethod
    def code_forward(self, batch_data: Dict, ) -> Any:
        pass

    @abc.abstractmethod
    def comment_forward(self, batch_data: Dict, ) -> Any:
        pass

    @abc.abstractmethod
    def train_pipeline(self, batch_data: Dict, ) -> Any:
        pass

    @abc.abstractmethod
    def eval_pipeline(self, batch_data: Dict, ) -> Any:
        pass

    def forward(self, cod_batch_data: Dict, cmt_batch_data: Dict, ) -> Any:
        cod_emb = self.code_encoder(cod_batch_data)
        cmt_emb = self.comment_encoder(cmt_batch_data)
        return cod_emb, cmt_emb,

    @abc.abstractmethod
    def train_sl(self, *args: Any, **kwargs: Any, ) -> Any:
        pass

    def __str__(self):
        return '{}(\n{}\n{}\n)'.format(
            self.__class__.__name__,
            self.code_encoder,
            self.comment_encoder,
        )
