# -*- coding: utf-8 -*-
from ncc import LOGGER
from ncc.model.template import *
from ncc.module.code2vec.multi_modal import *
from ncc.module.summarization import *
from ncc.metric import *
from typing import Dict, Tuple, Any


class AstAttendGru(Encoder2Decoder):

    def __init__(self, args: Dict,dict_comment ) -> None:
        LOGGER.debug('building {}...'.format(self.__class__.__name__))
        super(AstAttendGru, self).__init__(
            encoder=AstAttendGruEncoder(args),
            decoder=AstAttendGruDecoder(args, dict_comment.size),
        )
        self.args = args

    def eval_pipeline(self, batch : Dict, ) -> Tuple:
        tok_output, tok_enc_hc, sbtao_output, sbtao_enc_hc = self.encoder.forward(batch)
        comment_pred, comment_logprobs  = self.decoder.sample( batch, tok_output, tok_enc_hc, sbtao_output )

        # code_batch, _, code_padding_mask = batch['tok']
        # _, comment_input, _, _, _ = batch['comment']
        # enc_out = self.encoder(code_batch)
        # sample_opt = {'beam_size': 1, 'sample_max': 1, 'seq_length': self.max_predict_length}
        # comment_pred, comment_logprobs = self.decoder.sample(comment_input, enc_out, code_padding_mask, s_t=None,
        #                                                            sample_opt=sample_opt)

        return comment_pred, comment_logprobs
    #
    def train_sl(self, batch: Dict, criterion: BaseLoss, ) -> Any:

        tok_output, tok_enc_hc, sbtao_output, sbtao_enc_hc = self.encoder.forward(batch)
        comment_pred, comment_logprobs  = self.decoder.forward(batch, tok_output, tok_enc_hc, sbtao_output )
        _, comment_input, comment_target_ori, comment_len, raw_comment = batch['comment']
        # comment_target = comment_target_ori[:, :self.max_predict_length ]

        # LOGGER.info("comment_logprobs.shape:{} comment_target_ori.shape:{}".format(
        #     comment_logprobs.shape , comment_target_ori.shape))
        loss = criterion(comment_logprobs, comment_target_ori  )


        return loss


class AstAttendGruV2(Encoder2Decoder):
    def __init__(self, args: Dict,dict_comment ) -> None:
        LOGGER.debug('building {}...'.format(self.__class__.__name__))
        super(AstAttendGruV2, self).__init__(
            encoder=AstAttendGruV2Encoder(args),
            decoder=AstAttendGruDecoder(args, dict_comment.size),
        )
        self.args = args

    def eval_pipeline(self, batch : Dict, ) -> Tuple:
        tok_output, tok_enc_hc, sbtao_output, sbtao_enc_hc = self.encoder.forward(batch)
        comment_pred, comment_logprobs  = self.decoder.sample( batch, tok_output, tok_enc_hc, sbtao_output )

        # code_batch, _, code_padding_mask = batch['tok']
        # _, comment_input, _, _, _ = batch['comment']
        # enc_out = self.encoder(code_batch)
        # sample_opt = {'beam_size': 1, 'sample_max': 1, 'seq_length': self.max_predict_length}
        # comment_pred, comment_logprobs = self.decoder.sample(comment_input, enc_out, code_padding_mask, s_t=None,
        #                                                            sample_opt=sample_opt)

        return comment_pred, comment_logprobs
    #
    def train_sl(self, batch: Dict, criterion: BaseLoss, ) -> Any:

        tok_output, tok_enc_hc, sbtao_output, sbtao_enc_hc = self.encoder.forward(batch)
        comment_pred, comment_logprobs  = self.decoder.forward(batch, tok_output, tok_enc_hc, sbtao_output )
        _, comment_input, comment_target_ori, comment_len, raw_comment = batch['comment']
        # comment_target = comment_target_ori[:, :self.max_predict_length ]

        # LOGGER.info("comment_logprobs.shape:{} comment_target_ori.shape:{}".format(
        #     comment_logprobs.shape , comment_target_ori.shape))
        loss = criterion(comment_logprobs, comment_target_ori  )


        return loss

class AstAttendGruV3(Encoder2Decoder):
    def __init__(self, args: Dict,dict_comment ) -> None:
        LOGGER.debug('building {}...'.format(self.__class__.__name__))
        super(AstAttendGruV3, self).__init__(
            encoder=AstAttendGruV2Encoder(args),
            decoder=AstAttendGruV3Decoder(args, dict_comment.size),
        )
        self.args = args
        # self.max_predict_length = self.args['training']['max_predict_length']

    def eval_pipeline(self, batch : Dict, ) -> Tuple:
        tok_output, tok_enc_hc, sbtao_output, sbtao_enc_hc = self.encoder.forward(batch)
        comment_pred, comment_logprobs  = self.decoder.sample( batch, tok_output, tok_enc_hc, sbtao_output )

        # code_batch, _, code_padding_mask = batch['tok']
        # _, comment_input, _, _, _ = batch['comment']
        # enc_out = self.encoder(code_batch)
        # sample_opt = {'beam_size': 1, 'sample_max': 1, 'seq_length': self.max_predict_length}
        # comment_pred, comment_logprobs = self.decoder.sample(comment_input, enc_out, code_padding_mask, s_t=None,
        #                                                            sample_opt=sample_opt)

        return comment_pred, comment_logprobs
    #
    def train_sl(self, batch: Dict, criterion: BaseLoss, ) -> Any:

        tok_output, tok_enc_hc, sbtao_output, sbtao_enc_hc = self.encoder.forward(batch)
        comment_pred, comment_logprobs  = self.decoder.forward(batch, tok_output, tok_enc_hc, sbtao_output )
        _, comment_input, comment_target_ori, comment_len, raw_comment = batch['comment']
        # comment_target = comment_target_ori[:, :self.max_predict_length ]

        # LOGGER.info("comment_logprobs.shape:{} comment_target_ori.shape:{}".format(
        #     comment_logprobs.shape , comment_target_ori.shape))
        loss = criterion(comment_logprobs, comment_target_ori  )


        return loss

class AstAttendGruV4(Encoder2Decoder):
    def __init__(self, args: Dict,dict_comment ) -> None:
        LOGGER.debug('building {}...'.format(self.__class__.__name__))
        super(AstAttendGruV4, self).__init__(
            encoder=AstAttendGruV4Encoder(args),
            decoder=AstAttendGruV3Decoder(args, dict_comment.size),
        )
        self.args = args

    def eval_pipeline(self, batch: Dict, ) -> Tuple:
        tok_output, tok_enc_hc, sbtao_output = self.encoder.forward(batch)
        comment_pred, comment_logprobs  = self.decoder.sample( batch, tok_output, tok_enc_hc, sbtao_output )

        # code_batch, _, code_padding_mask = batch['tok']
        # _, comment_input, _, _, _ = batch['comment']
        # enc_out = self.encoder(code_batch)
        # sample_opt = {'beam_size': 1, 'sample_max': 1, 'seq_length': self.max_predict_length}
        # comment_pred, comment_logprobs = self.decoder.sample(comment_input, enc_out, code_padding_mask, s_t=None,
        #                                                            sample_opt=sample_opt)

        return comment_pred, comment_logprobs
    #
    def train_sl(self, batch: Dict, criterion: BaseLoss, ) -> Any:

        tok_output, tok_enc_hc, sbtao_output = self.encoder.forward(batch)
        comment_pred, comment_logprobs  = self.decoder.forward(batch, tok_output, tok_enc_hc, sbtao_output )
        _, comment_input, comment_target_ori, comment_len, raw_comment = batch['comment']
        # comment_target = comment_target_ori[:, :self.max_predict_length ]

        # LOGGER.info("comment_logprobs.shape:{} comment_target_ori.shape:{}".format(
        #     comment_logprobs.shape , comment_target_ori.shape))
        loss = criterion(comment_logprobs, comment_target_ori  )


        return loss