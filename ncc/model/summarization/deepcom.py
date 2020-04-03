# -*- coding: utf-8 -*-
from ncc import LOGGER
from ncc.model.template import Encoder2Decoder
from ncc.module.code2vec.encoder_tok import DeepComEncoder_EmbRNN
from ncc.module.summarization import SeqDecoder
from ncc.metric import BaseLoss
from typing import Dict, Any, Tuple

class DeepCom(Encoder2Decoder):
    def __init__(self, args: Dict) -> None:
        LOGGER.debug('building {}...'.format(self.__class__.__name__))
        super(DeepCom, self).__init__(
            encoder=DeepComEncoder_EmbRNN.load_from_config(args),
            decoder=SeqDecoder.load_from_config(args, modal='comment'),
        )
        self.args = args

    def eval_pipeline(self, batch_data: Dict, ) -> Tuple:
        # train/eval pipeline may be quite different, therefore we design two methods
        enc_output, dec_hidden, enc_mask = self.encoder.forward(batch_data)
        sample_opt = {'beam_size': 1, 'sample_max': 1, 'seq_length': self.args['training']['max_predict_length']}
        comment_pred, comment_logprobs, _, _, = \
            self.decoder.sample(batch_data, enc_output, dec_hidden, enc_mask, sample_opt)
        return comment_pred, comment_logprobs,

    def train_sl(self, batch: Dict, criterion: BaseLoss, ) -> Any:
        # _, comment_logprobs, _, _, _, = self.train_pipeline(batch)
        enc_output, dec_hidden, enc_mask = self.encoder.forward(batch)
        # LOGGER.info(enc_output.keys())
        sample_opt = {'sample_max': 1, 'seq_length': self.args['training']['max_predict_length']}
        _, comment_logprobs, _, _, _, _, _, = self.decoder.forward(batch, enc_output, dec_hidden, enc_mask, sample_opt)

        if self.args['training']['pointer']:
            comment_target = batch['pointer'][1][:, :self.args['training']['max_predict_length']]
        else:
            comment_target = batch['comment'][2][:, :self.args['training']['max_predict_length']]
        # print('comment_logprobs: ', comment_logprobs.size())
        # print('comment_target_batch2use: ', comment_target_batch2use.size())

        loss = criterion(comment_logprobs, comment_target)
        # print('loss: ', loss.item())
        return loss
