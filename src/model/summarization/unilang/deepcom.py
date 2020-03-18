# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

from src import *
from src.eval import *
from src.model.template import *
from src.module.code2vec.multi_modal import *
from src.module.code2vec.encoder_tok import *
from src.module.summarization import *
from src.model import *
from src.dataset import *
from src.metric import *
from src.utils.util_data import batch_to_cuda
from src.data import *


class DeepCom(Encoder2Decoder):

    def __init__(self, config: Dict) -> None:
        LOGGER.debug('building {}...'.format(self.__class__.__name__))
        super(DeepCom, self).__init__(
            # encoder=MMEncoder_EmbRNN.load_from_config(config),
            encoder=DeepComEncoder_EmbRNN.load_from_config(config),
            decoder=SeqDecoder.load_from_config(config, modal='comment'),
        )
        self.config = config

    def eval_pipeline(self, batch_data: Dict, ) -> Tuple:
        # train/eval pipeline may be quite different, therefore we design two methods
        enc_output, dec_hidden, enc_mask = self.encoder.forward(batch_data)
        sample_opt = {'beam_size': 1, 'sample_max': 1, 'seq_length': self.config['training']['max_predict_length']}
        comment_pred, comment_logprobs, _, _, = \
            self.decoder.sample(batch_data, enc_output, dec_hidden, enc_mask, sample_opt)
        return comment_pred, comment_logprobs,

    def train_sl(self, batch: Dict, criterion: BaseLoss, ) -> Any:
        # _, comment_logprobs, _, _, _, = self.train_pipeline(batch)
        enc_output, dec_hidden, enc_mask = self.encoder.forward(batch)
        # LOGGER.info(enc_output.keys())
        sample_opt = {'sample_max': 1, 'seq_length': self.config['training']['max_predict_length']}
        _, comment_logprobs, _, _, _, _, _, = self.decoder.forward(batch, enc_output, dec_hidden, enc_mask, sample_opt)

        if self.config['training']['pointer']:
            comment_target = batch['pointer'][1][:, :self.config['training']['max_predict_length']]
        else:
            comment_target = batch['comment'][2][:, :self.config['training']['max_predict_length']]
        # print('comment_logprobs: ', comment_logprobs.size())
        # print('comment_target_batch2use: ', comment_target_batch2use.size())

        loss = criterion(comment_logprobs, comment_target)
        # print('loss: ', loss.item())
        return loss
