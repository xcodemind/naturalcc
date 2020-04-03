# -*- coding: utf-8 -*-
from ncc import LOGGER
from ncc.model.template import Encoder2Decoder
from ncc.module.code2vec.encoder_ast import Encoder_EmbPathRNN
from ncc.module.summarization import SeqDecoder
from ncc.metric import BaseLoss
from typing import Any, Dict, Tuple


class Code2Seq(Encoder2Decoder):

    def __init__(self, args: Dict) -> None:
        LOGGER.debug('building {}...'.format(self.__class__.__name__))
        super(Code2Seq, self).__init__(
            encoder=Encoder_EmbPathRNN.load_from_args(args),
            decoder=SeqDecoder.load_from_args(args, 'comment'),
        )
        self.args = args

    def eval_pipeline(self, batch_data: Dict, ) -> Tuple:
        # train/eval pipeline may be quite different, therefore we design two methods
        enc_output, dec_hidden, enc_mask = self.encoder.forward(*batch_data['path'])
        enc_output = {'path': enc_output}
        enc_mask = {'path': enc_mask}

        sample_opt = {'beam_size': 1, 'sample_max': 1, 'seq_length': self.args['training']['max_predict_length']}
        comment_pred, comment_logprobs, _, _, = \
            self.decoder.sample(batch_data, enc_output, dec_hidden, enc_mask, sample_opt)
        return comment_pred, comment_logprobs,

    def train_sl(self, batch_data: Dict, criterion: BaseLoss, ) -> Any:
        enc_output, dec_hidden, enc_mask = self.encoder.forward(*batch_data['path'])
        enc_output = {'path': enc_output}
        enc_mask = {'path': enc_mask}
        sample_opt = {'sample_max': 1, 'seq_length': self.args['training']['max_predict_length']}
        _, comment_logprobs, _, _, _, _, _, = \
            self.decoder.forward(batch_data, enc_output, dec_hidden, enc_mask, sample_opt)

        if self.args['training']['pointer']:
            comment_target_batch2use = batch_data['pointer'][1]
        else:
            comment_target_batch2use = batch_data['comment'][2]
        loss = criterion(comment_logprobs, comment_target_batch2use)
        return loss
