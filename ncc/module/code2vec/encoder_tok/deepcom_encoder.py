# -*- coding: utf-8 -*-
import torch.nn as nn
from torch.nn import Module
from ncc import LOGGER
from ncc.module.code2vec.encoder_tok import *
from typing import Dict, Any


class DeepComEncoder_EmbRNN(Module):
    def __init__(self, args: Dict, ):
        super(DeepComEncoder_EmbRNN, self).__init__()
        self.args = args
        modality_count = 0
        LOGGER.info("init {}".format(self.__class__.__name__))
        LOGGER.debug("code_modalities: {}".format(self.args['training']['code_modalities']))
        if ('tok' in self.args['training']['code_modalities']) or \
                ('sbt' in self.args['training']['code_modalities']):
            self.tok_encoder = Encoder_EmbRNN.load_from_config(args, 'tok') \
                if 'tok' in self.args['training']['code_modalities'] else \
                Encoder_EmbRNN.load_from_config(args, 'sbt')
            if self.args['training']['code_modal_transform']:
                self.transform_tok = nn.Linear(self.args['training']['rnn_hidden_size'],
                                               self.args['training']['rnn_hidden_size'])
            modality_count += 1

        LOGGER.debug('modality_count: {}'.format(modality_count))


    @classmethod
    def load_from_config(cls, args: Dict) -> Any:
        instance = cls(
            args=args,
        )
        return instance


    def forward(self, batch) -> Any:
        enc_output, enc_hidden_state, enc_mask = {}, {}, {}
        if ('tok' in self.args['training']['code_modalities']) or \
                ('sbt' in self.args['training']['code_modalities']):
            code_batch, code_length, code_padding_mask = batch['tok']
            tok_enc_hc = self.tok_encoder.init_hidden(code_batch.size(0))
            # (batch_size*maxL*rnn_hidden_size, (1*batch_size*rnn_hidden_size, 1*batch_size*rnn_hidden_size))
            tok_output, tok_enc_hc = self.tok_encoder.forward(code_batch, code_length, tok_enc_hc)
            enc_output['tok'], enc_hidden_state['tok'], enc_mask['tok'] = tok_output, tok_enc_hc, code_padding_mask

        dec_hc = enc_hidden_state['tok']
        return enc_output, dec_hc, enc_mask
