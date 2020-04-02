# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import Module
from ncc import LOGGER
from ncc.module.code2vec.encoder_tok import Encoder_EmbRNN
from ncc.module.code2vec.encoder_ast import Encoder_EmbTreeRNN, Encoder_EmbPathRNN
from ncc.module.attention import HirarchicalAttention
from typing import Dict, Any


class CodeEnocder_MM(Module):
    '''
    Multi-modalities code encoder
    emb -> RNN -> Hierarchical Attention -> concate -> linear
    '''

    def __init__(self, config):
        super(CodeEnocder_MM, self).__init__()
        self.config = config
        self.code_modalities = self.config['training']['code_modalities']
        modality_count = len(self.code_modalities)
        for modal in self.code_modalities:
            # RNN ->  Hierarchical Attention
            self._build_modal_pipeline(modal)
        # fusion layer
        self.fuse_linear = nn.Linear(self.config['training']['rnn_hidden_size'] * modality_count, \
                                     self.config['training']['rnn_hidden_size'])

    def _build_modal_pipeline(self, modal: str):
        if 'tok' == modal:
            encoder = Encoder_EmbRNN.load_from_config(self.config, modal='tok')
        elif 'ast' == modal:
            encoder = Encoder_EmbTreeRNN.load_from_config(self.config, modal='ast')
        elif 'path' == modal:
            encoder = Encoder_EmbPathRNN.load_from_config(self.config)
        else:
            raise NotImplementedError('no such {} modal'.format(modal))
        self.__setattr__(name='{}_encoder'.format(modal), value=encoder)
        rnn_bidirectional = self.config['training']['rnn_bidirectional']
        if self.config['training']['attn_type'] == 'hier':
            attn_layer = HirarchicalAttention(
                self.config['training']['rnn_hidden_size'] * (2 if rnn_bidirectional else 1))
            self.__setattr__(name='{}_attn'.format(modal), value=attn_layer)
        else:
            pass

    def forward(self, batch: Dict, ) -> Any:
        hidden = {}
        if 'tok' in self.code_modalities:
            tok_batch, tok_length, tok_mask = batch['tok']
            # LSTM
            tok_feature, tok_hc = self.tok_encoder(tok_batch, tok_length, )
            LOGGER.debug('tok feature: {}'.format(tok_feature.size()))
            LOGGER.debug('tok hidden: {} {}'.format(tok_hc[0].size(), tok_hc[1].size()))
            if self.config['training']['attn_type'] == 'hier':
                # Hierarchical Attention
                tok_feature = self.tok_attn(tok_feature, )
                LOGGER.debug('tok feature after attn: {}'.format(tok_feature.size()))
            else:
                tok_feature, _ = tok_feature.max(dim=1)
                LOGGER.debug('tok feature after attn: {}'.format(tok_feature.size()))
            hidden['tok'] = tok_feature

        if 'ast' in self.code_modalities:
            ast_dgl_batch, ast_dgl_root_index, ast_dgl_node_num, ast_mask = batch['ast']
            # LSTM
            enc_hc = self.ast_encoder.init_hidden(ast_dgl_batch.graph.number_of_nodes())
            ast_feature, enc_hc = self.ast_encoder(ast_dgl_batch, enc_hc, ast_dgl_root_index, ast_dgl_node_num)
            LOGGER.debug('ast feature: {}'.format(ast_feature.size()))
            LOGGER.debug('ast hidden: {} {}'.format(enc_hc[0].size(), enc_hc[1].size()))
            if self.config['training']['attn_type'] is not None:
                # Hierarchical Attention
                ast_feature = self.ast_attn(ast_feature)
                LOGGER.debug('ast feature after attn: {}'.format(ast_feature.size()))
            hidden['ast'] = ast_feature

        if 'path' in self.code_modalities:
            path_feature, enc_hc, _ = self.path_encoder(*batch['path'])
            LOGGER.debug('path feature: {}'.format(path_feature.size()))
            LOGGER.debug('path hidden: {} {}'.format(enc_hc[0].size(), enc_hc[1].size()))
            if self.config['training']['attn_type'] == 'hier':
                # Hierarchical Attention
                path_feature = self.path_attn(path_feature)
                LOGGER.debug('path feature after attn: {}'.format(path_feature.size()))
            else:
                path_feature, _ = path_feature.max(dim=1)
                LOGGER.debug('path feature after attn: {}'.format(path_feature.size()))
            hidden['path'] = path_feature

        if len(self.code_modalities) == 1:
            code_feature = hidden[self.code_modalities[0]]
        else:  # len(self.code_modalities) > 1
            all_hidden = torch.cat([hidden[modal] for modal in self.code_modalities], dim=-1)
            LOGGER.debug('concate {}'.format(all_hidden.size()))
            code_feature = self.fuse_linear(all_hidden)
            LOGGER.debug('fuse: {}'.format(code_feature.size()))
        code_feature = torch.tanh(code_feature)
        return code_feature
