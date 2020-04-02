# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from ncc import LOGGER
from ncc.module.code2vec.encoder_tok import *
from ncc.module.code2vec.encoder_ast import *
from typing import Dict, Any, Tuple

class MMEncoder_EmbRNN(Module):
    def __init__(self, config: Dict, ):
        super(MMEncoder_EmbRNN, self).__init__()
        self.config = config
        modality_count = 0
        # print('config-: ', config)
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

        if 'ast' in self.config['training']['code_modalities']:
            self.tree_encoder = Encoder_EmbTreeRNN.load_from_config(config, 'ast')
            if self.config['training']['code_modal_transform']:
                self.transform_tree = nn.Linear(self.config['training']['rnn_hidden_size'],
                                                self.config['training']['rnn_hidden_size'])
            modality_count += 1

        if 'path' in self.config['training']['code_modalities']:
            self.path_encoder = Encoder_EmbPathRNN.load_from_config(config, )
            if self.config['training']['code_modal_transform']:
                self.transform_tree = nn.Linear(self.config['training']['rnn_hidden_size'],
                                                self.config['training']['rnn_hidden_size'])
            modality_count += 1

        if 'cfg' in self.config['training']['code_modalities']:
            raise NotImplementedError
        LOGGER.debug('modality_count: {}'.format(modality_count))
        # fuse code modalities with a fc
        if modality_count > 1:
            self.fuse = nn.Linear(self.config['training']['rnn_hidden_size'] * modality_count,
                                  self.config['training']['rnn_hidden_size'])

        if self.config['training']['enc_hc2dec_hc'] == 'hc':
            self.fuse_c = nn.Linear(self.config['training']['rnn_hidden_size'] * modality_count,
                                    self.config['training']['rnn_hidden_size'])

    @classmethod
    def load_from_config(cls, config: Dict) -> Any:
        instance = cls(
            config=config,
        )
        return instance

    def _enc_hc2dec_hc(self, enc_hc: Dict) -> Tuple:
        dec_hc = {
            'h': [],
            'c': [],
        }

        # fuse all hc of different modalties
        if self.config['training']['code_modal_transform']:
            if 'h' in self.config['training']['enc_hc2dec_hc']:
                if ('tok' in enc_hc) or ('sbt' in enc_hc):
                    tok_feat = torch.tanh(
                        self.transform_tok(
                            F.dropout(enc_hc['tok'][0], self.config['training']['dropout'], training=self.training))
                    ).reshape(1, -1, self.config['training']['rnn_hidden_size'])
                    dec_hc['h'].append(tok_feat)
                if 'ast' in enc_hc:
                    tree_feat = torch.tanh(
                        self.transform_tree(
                            F.dropout(enc_hc['ast'][0], self.config['training']['dropout'], training=self.training))
                    ).reshape(1, -1, self.config['training']['rnn_hidden_size'])
                    dec_hc['h'].append(tree_feat)
                if 'path' in enc_hc:
                    path_feat = torch.tanh(
                        self.transform_tree(
                            F.dropout(enc_hc['path'][0], self.config['training']['dropout'], training=self.training))
                    ).reshape(1, -1, self.config['training']['rnn_hidden_size'])
                    dec_hc['h'].append(path_feat)
            if 'c' in self.config['training']['enc_hc2dec_hc']:
                if ('tok' in enc_hc) or ('sbt' in enc_hc):
                    tok_feat = torch.tanh(
                        self.transform_tok(
                            F.dropout(enc_hc['tok'][1], self.config['training']['dropout'], training=self.training))
                    ).reshape(1, -1, self.config['training']['rnn_hidden_size'])
                    dec_hc['c'].append(tok_feat)
                if 'ast' in enc_hc:
                    tree_feat = torch.tanh(
                        self.transform_tree(
                            F.dropout(enc_hc['ast'][1], self.config['training']['dropout'], training=self.training))
                    ).reshape(1, -1, self.config['training']['rnn_hidden_size'])
                    # print('tree_feat: ', tree_feat.size())
                    # print(tree_feat)
                    dec_hc['c'].append(tree_feat)
                if 'path' in enc_hc:
                    path_feat = torch.tanh(
                        self.transform_tok(
                            F.dropout(enc_hc['path'][1], self.config['training']['dropout'], training=self.training))
                    ).reshape(1, -1, self.config['training']['rnn_hidden_size'])
                    dec_hc['c'].append(path_feat)

        else:
            if 'h' in self.config['training']['enc_hc2dec_hc']:
                if ('tok' in enc_hc) or ('sbt' in enc_hc):
                    dec_hc['h'].append(enc_hc['tok'][0].reshape(1, -1, self.config['training']['rnn_hidden_size']))
                if 'ast' in enc_hc:
                    dec_hc['h'].append(enc_hc['ast'][0].reshape(1, -1, self.config['training']['rnn_hidden_size']))
                if 'path' in enc_hc:
                    dec_hc['h'].append(enc_hc['path'][0].reshape(1, -1, self.config['training']['rnn_hidden_size']))
            if 'c' in self.config['training']['enc_hc2dec_hc']:
                if ('tok' in enc_hc) or ('sbt' in enc_hc):
                    dec_hc['c'].append(enc_hc['tok'][1].reshape(1, -1, self.config['training']['rnn_hidden_size']))
                if 'ast' in enc_hc:
                    dec_hc['c'].append(enc_hc['ast'][1].reshape(1, -1, self.config['training']['rnn_hidden_size']))
                if 'path' in enc_hc:
                    dec_hc['c'].append(enc_hc['path'][1].reshape(1, -1, self.config['training']['rnn_hidden_size']))

        if len(dec_hc['h']) > 1:
            dec_hc['h'] = torch.cat(dec_hc['h'], 2)
            dec_hc['h'] = torch.tanh(
                self.fuse(F.dropout(dec_hc['h'], self.config['training']['dropout'], training=self.training))
            ).reshape(1, -1, self.config['training']['rnn_hidden_size'])
        elif len(dec_hc['h']) == 1:
            dec_hc['h'] = dec_hc['h'][0]
        else:
            raise NotImplementedError

        if len(dec_hc['c']) > 1:
            dec_hc['c'] = torch.cat(dec_hc['c'], 2)
            dec_hc['c'] = torch.tanh(
                self.fuse_c(F.dropout(dec_hc['c'], self.config['training']['dropout'], training=self.training))
            ).reshape(1, -1, self.config['training']['rnn_hidden_size'])
        elif len(dec_hc['c']) == 1:
            dec_hc['c'] = dec_hc['c'][0]
        else:
            dec_hc['c'] = torch.zeros_like(dec_hc['h'])

        return dec_hc['h'], dec_hc['c'],

    def forward(self, batch) -> Any:
        enc_output, enc_hidden_state, enc_mask = {}, {}, {}
        if ('tok' in self.config['training']['code_modalities']) or \
                ('sbt' in self.config['training']['code_modalities']):
            code_batch, code_length, code_padding_mask = batch['tok']
            tok_enc_hc = self.tok_encoder.init_hidden(code_batch.size(0))
            # (batch_size*maxL*rnn_hidden_size, (1*batch_size*rnn_hidden_size, 1*batch_size*rnn_hidden_size))
            tok_output, tok_enc_hc = self.tok_encoder.forward(code_batch, code_length, tok_enc_hc)
            enc_output['tok'], enc_hidden_state['tok'], enc_mask['tok'] = tok_output, tok_enc_hc, code_padding_mask

        if 'ast' in self.config['training']['code_modalities']:
            tree_dgl_batch, tree_dgl_root_index, tree_dgl_node_num, tree_padding_mask = batch['ast']
            tree_enc_hidden = self.tree_encoder.init_hidden(tree_dgl_batch.graph.number_of_nodes())
            tree_output, enc_hc = self.tree_encoder.forward(tree_dgl_batch, tree_enc_hidden, tree_dgl_root_index,
                                                            tree_dgl_node_num)
            enc_output['ast'], enc_hidden_state['ast'], enc_mask['ast'] = tree_output, enc_hc, tree_padding_mask

        if 'path' in self.config['training']['code_modalities']:
            enc_output['path'], enc_hidden_state['path'], enc_mask['path'] = \
                self.path_encoder.forward(*batch['path'])

        if self.config['training']['enc_hc2dec_hc'] is None:
            dec_hc = None
        else:
            dec_hc = self._enc_hc2dec_hc(enc_hidden_state)
        return enc_output, dec_hc, enc_mask
