# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from ncc import *

from ncc.module.code2vec.encoder_tok import *
from ncc.module.code2vec.encoder_ast import *


class DeepComEncoder_EmbRNN(Module):
    def __init__(self, config: Dict, ):
        super(DeepComEncoder_EmbRNN, self).__init__()
        self.config = config
        modality_count = 0
        # print('config-: ', config)
        LOGGER.info("init {}".format(self.__class__.__name__))
        LOGGER.debug("code_modalities: {}".format(self.config['training']['code_modalities']))
        if ('tok' in self.config['training']['code_modalities']) or \
                ('sbt' in self.config['training']['code_modalities']):
            self.tok_encoder = Encoder_EmbRNN.load_from_config(config, 'tok') \
                if 'tok' in self.config['training']['code_modalities'] else \
                Encoder_EmbRNN.load_from_config(config, 'sbt')
            if self.config['training']['code_modal_transform']:
                self.transform_tok = nn.Linear(self.config['training']['rnn_hidden_size'],
                                               self.config['training']['rnn_hidden_size'])
            modality_count += 1

        LOGGER.debug('modality_count: {}'.format(modality_count))
        # fuse code modalities with a fc
        # if modality_count > 1:
        #     self.fuse = nn.Linear(self.config['training']['rnn_hidden_size'] * modality_count,
        #                           self.config['training']['rnn_hidden_size'])
        #
        # if self.config['training']['enc_hc2dec_hc'] == 'hc':
        #     self.fuse_c = nn.Linear(self.config['training']['rnn_hidden_size'] * modality_count,
        #                             self.config['training']['rnn_hidden_size'])

    @classmethod
    def load_from_config(cls, config: Dict) -> Any:
        instance = cls(
            config=config,
        )
        return instance



    def forward(self, batch) -> Any:
        enc_output, enc_hidden_state, enc_mask = {}, {}, {}
        if ('tok' in self.config['training']['code_modalities']) or \
                ('sbt' in self.config['training']['code_modalities']):
            code_batch, code_length, code_padding_mask = batch['tok']
            tok_enc_hc = self.tok_encoder.init_hidden(code_batch.size(0))
            # (batch_size*maxL*rnn_hidden_size, (1*batch_size*rnn_hidden_size, 1*batch_size*rnn_hidden_size))
            tok_output, tok_enc_hc = self.tok_encoder.forward(code_batch, code_length, tok_enc_hc)
            enc_output['tok'], enc_hidden_state['tok'], enc_mask['tok'] = tok_output, tok_enc_hc, code_padding_mask




        dec_hc = enc_hidden_state['tok']
        return enc_output, dec_hc, enc_mask
