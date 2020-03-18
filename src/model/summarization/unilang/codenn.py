# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

from src import *
from src.model.template import *
from src.module.code2vec.base import *
from src.module.summarization import *
from src.metric import *



class CodeNN(Encoder2Decoder):

    def __init__(self, config: Dict,dict_comment) -> None:
        LOGGER.debug('building {}...'.format(self.__class__.__name__))
        super(CodeNN, self).__init__(
            encoder=Encoder_Emb(token_num=config['training']['token_num']['tok'],
                                embed_size=400 ),
            decoder=CodeNNSeqDecoder(config['training']['max_predict_length'],dict_comment),
        )
        self.config = config
        self.max_predict_length = self.config['training']['max_predict_length']

    def eval_pipeline(self, batch : Dict, ) -> Tuple:

## codenn
        code_batch, _, code_padding_mask = batch['tok']
        _, comment_input, _, _, _ = batch['comment']
        enc_out = self.encoder(code_batch)
        sample_opt = {'beam_size': 1, 'sample_max': 1, 'seq_length': self.max_predict_length}
        comment_pred, comment_logprobs = self.decoder.sample(comment_input, enc_out, code_padding_mask, s_t=None,
                                                                   sample_opt=sample_opt)
        # comment_loss = lm_criterion(comment_logprobs, comment_target_batch)
###

        return comment_pred, comment_logprobs#, comment_target_padded,

    def train_sl(self, batch: Dict, criterion: BaseLoss, ) -> Any:

        code_batch, code_length, code_padding_mask = batch['tok']
        _, comment_input, comment_target_ori, comment_len, raw_comment = batch['comment']
        enc_out = self.encoder(code_batch)
        _, comment_logprobs = self.decoder.forward(comment_input, enc_out, code_padding_mask, s_t=None,
                                                               sample_opt={})

        comment_target = comment_target_ori[:, :self.max_predict_length ]

        # LOGGER.debug("comment_target_ori.shape: {}".format(comment_target_ori.shape))
        # LOGGER.debug("comment_target.shape: {}".format(comment_target.shape))
        # LOGGER.debug("comment_logprobs.shape: {}".format(comment_logprobs.shape))
        loss = criterion(comment_logprobs, comment_target  )


        return loss


