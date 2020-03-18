# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from src import *

from src.module.code2vec.encoder_tok import *
from src.module.code2vec.encoder_ast import *


class AstAttendGruEncoder(Module):
    def __init__(self, config: Dict ):
        super(AstAttendGruEncoder, self).__init__()
        self.config = config


        LOGGER.debug("code_modalities: {}".format(self.config['training']['code_modalities']))
        assert set(self.config['training']['code_modalities']) == {'tok','sbtao'}

        # self.tok_encoder = Encoder_EmbRNN(
        #     token_num=config['training']['token_num']['tok'],
        #     embed_size= 100 ,
        #     rnn_type= 'GRU',
        #     hidden_size= 256 ,
        #     layer_num=  1 ,
        #     dropout= 0 ,
        #     bidirectional= False )
        # self.sbtao_encoder = Encoder_EmbRNN(
        #     token_num=config['training']['token_num']['sbtao'],
        #     embed_size= 10 ,
        #     rnn_type= 'GRU',
        #     hidden_size= 256 ,
        #     layer_num=  1 ,
        #     dropout= 0 ,
        #     bidirectional= False )


        self.tok_emb = torch.nn.Embedding(num_embeddings=config['training']['token_num']['tok'],
                                          embedding_dim=100, padding_idx=self.config['training']['padding_idx'],
                                         max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False)

        self.tok_encoder = torch.nn.GRU(  input_size=100, hidden_size=256 ,  num_layers=1, bias= True,
                                          batch_first=True,    dropout=0,    bidirectional=False)

        self.sbtao_emb = torch.nn.Embedding(num_embeddings=config['training']['token_num']['sbtao'],
                                          embedding_dim=10 , padding_idx= self.config['training']['padding_idx'],
                                         max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False)

        self.sbtao_encoder = torch.nn.GRU(  input_size=10, hidden_size=256 ,  num_layers=1, bias= True,
                                          batch_first=True,    dropout=0,    bidirectional=False)


    def forward(self, batch) -> Any:


        sbtao_batch, sbtao_length, sbtao_padding_mask = batch['sbtao']
        sbtao_batch  = self.sbtao_emb(sbtao_batch)
        sbtao_output, sbtao_enc_hc = self.sbtao_encoder.forward(sbtao_batch )

        code_batch, code_length, code_padding_mask = batch['tok']
        code_batch = self.tok_emb(code_batch)
        tok_output, tok_enc_hc = self.tok_encoder.forward(code_batch)

        return tok_output, tok_enc_hc, sbtao_output, sbtao_enc_hc

class AstAttendGruV2Encoder(Module):
    def __init__(self, config: Dict ):
        super(AstAttendGruV2Encoder, self).__init__()
        self.config = config


        LOGGER.debug("code_modalities: {}".format(self.config['training']['code_modalities']))
        assert set(self.config['training']['code_modalities']) == {'tok','sbtao'}

        self.tok_encoder = Encoder_EmbRNN(
            token_num=config['training']['token_num']['tok'],
            embed_size= 100 ,
            rnn_type= 'GRU',
            hidden_size= 256 ,
            layer_num=  1 ,
            dropout= 0 ,
            bidirectional= False )
        self.sbtao_encoder = Encoder_EmbRNN(
            token_num=config['training']['token_num']['sbtao'],
            embed_size= 10 ,
            rnn_type= 'GRU',
            hidden_size= 256 ,
            layer_num=  1 ,
            dropout= 0 ,
            bidirectional= False )


        # self.tok_emb = torch.nn.Embedding(num_embeddings=config['training']['token_num']['tok'],
        #                                   embedding_dim=100, padding_idx=self.config['training']['padding_idx'],
        #                                  max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False)
        #
        # self.tok_encoder = torch.nn.GRU(  input_size=100, hidden_size=256 ,  num_layers=1, bias= True,
        #                                   batch_first=True,    dropout=0,    bidirectional=False)
        #
        # self.sbtao_emb = torch.nn.Embedding(num_embeddings=config['training']['token_num']['sbtao'],
        #                                   embedding_dim=10 , padding_idx= self.config['training']['padding_idx'],
        #                                  max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False)
        #
        # self.sbtao_encoder = torch.nn.GRU(  input_size=10, hidden_size=256 ,  num_layers=1, bias= True,
        #                                   batch_first=True,    dropout=0,    bidirectional=False)


    def forward(self, batch) -> Any:
        sbtao_batch, sbtao_length, sbtao_padding_mask = batch['sbtao']
        sbtao_enc_hc = self.sbtao_encoder.init_hidden(sbtao_batch.size(0))
        # (batch_size*maxL*rnn_hidden_size, (1*batch_size*rnn_hidden_size, 1*batch_size*rnn_hidden_size))
        sbtao_output, sbtao_enc_hc = self.sbtao_encoder.forward(input = sbtao_batch, input_len=sbtao_length,
                                                                hidden= sbtao_enc_hc)

        code_batch, code_length, code_padding_mask = batch['tok']
        tok_enc_hc = self.tok_encoder.init_hidden(code_batch.size(0))
        # (batch_size*maxL*rnn_hidden_size, (1*batch_size*rnn_hidden_size, 1*batch_size*rnn_hidden_size))
        tok_output, tok_enc_hc = self.tok_encoder.forward(input = code_batch,input_len= code_length, hidden=tok_enc_hc)

        # sbtao_batch, sbtao_length, sbtao_padding_mask = batch['sbtao']
        # sbtao_batch  = self.sbtao_emb(sbtao_batch)
        # sbtao_output, sbtao_enc_hc = self.sbtao_encoder.forward(sbtao_batch )
        #
        # code_batch, code_length, code_padding_mask = batch['tok']
        # code_batch = self.tok_emb(code_batch)
        # tok_output, tok_enc_hc = self.tok_encoder.forward(code_batch)

        return tok_output, tok_enc_hc, sbtao_output, sbtao_enc_hc

class AstAttendGruV4Encoder(Module):
    def __init__(self, config: Dict ):
        super(AstAttendGruV4Encoder, self).__init__()
        self.config = config
        LOGGER.debug("code_modalities: {}".format(self.config['training']['code_modalities']))
        assert set(self.config['training']['code_modalities']) == {'tok','sbtao'}

        self.tok_encoder = Encoder_EmbRNN(
            token_num=config['training']['token_num']['tok'],
            embed_size= config['training']['tok_embed_size'] ,
            rnn_type= 'GRU',
            hidden_size= config['training']['rnn_hidden_size'] ,
            layer_num=  1 ,
            dropout= 0 ,
            bidirectional= False )
        self.sbtao_encoder = Encoder_EmbRNN(
            token_num=config['training']['token_num']['sbtao'],
            embed_size= config['training']['sbtao_embed_size'] ,
            rnn_type= 'GRU',
            hidden_size= config['training']['rnn_hidden_size']  ,
            layer_num=  1 ,
            dropout= 0 ,
            bidirectional= False )


    def forward(self, batch) -> Any:
        sbtao_batch, sbtao_length, sbtao_padding_mask = batch['sbtao']
        sbtao_enc_hc = self.sbtao_encoder.init_hidden(sbtao_batch.size(0))
        # (batch_size*maxL*rnn_hidden_size, (1*batch_size*rnn_hidden_size, 1*batch_size*rnn_hidden_size))
        sbtao_output, sbtao_enc_hc = self.sbtao_encoder.forward(input = sbtao_batch, input_len=sbtao_length,
                                                                hidden= sbtao_enc_hc)

        code_batch, code_length, code_padding_mask = batch['tok']
        # tok_enc_hc = self.tok_encoder.init_hidden(code_batch.size(0))
        # (batch_size*maxL*rnn_hidden_size, (1*batch_size*rnn_hidden_size, 1*batch_size*rnn_hidden_size))
        tok_output, tok_enc_hc = self.tok_encoder.forward(input = code_batch,input_len= code_length, hidden=sbtao_enc_hc)

        # sbtao_batch, sbtao_length, sbtao_padding_mask = batch['sbtao']
        # sbtao_batch  = self.sbtao_emb(sbtao_batch)
        # sbtao_output, sbtao_enc_hc = self.sbtao_encoder.forward(sbtao_batch )
        #
        # code_batch, code_length, code_padding_mask = batch['tok']
        # code_batch = self.tok_emb(code_batch)
        # tok_output, tok_enc_hc = self.tok_encoder.forward(code_batch)

        return tok_output, tok_enc_hc, sbtao_output