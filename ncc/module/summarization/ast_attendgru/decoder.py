# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from ncc import LOGGER
from ncc.module.code2vec.base import Encoder_Emb, Encoder_RNN
from ncc.utils.constants import *
from typing import Tuple

class AstAttendGruDecoder(nn.Module):
    def __init__(self, config,token_num) -> None:
        super(AstAttendGruDecoder, self).__init__()
        # embedding params
        self.config = config
        self.token_num = token_num
        self.embed_size = 100
        # rnn params
        self.hidden_size = 256
        self.layer_num = 1
        self.dropout =  0
        self.rnn_type = 'GRU'
        self.bidirectional = False
        self.max_comment_len =  config['dataset']['max_comment_len']
        # self.max_predict_length = max_predict_length  # decoder predict length


        # self.wemb = Encoder_Emb(self.token_num, self.embed_size, )
        # self.rnn = Encoder_RNN(self.rnn_type, self.embed_size, self.hidden_size, self.layer_num,
        #                        self.dropout, self.bidirectional)


        self.wemb = torch.nn.Embedding(num_embeddings=self.token_num ,
                                          embedding_dim=self.embed_size,
                                       padding_idx=self.config['training']['padding_idx'],
                                         max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False)

        self.rnn = torch.nn.GRU(  input_size=100, hidden_size=self.hidden_size ,  num_layers=self.layer_num,
                                  bias= True,  batch_first=True,
                                  dropout=self.dropout,  bidirectional=self.bidirectional)

        # self.concat_map = nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size)
        self.linear_1 = nn.Linear(self.hidden_size*3, self.hidden_size)
        self.linear_2 = nn.Linear(self.hidden_size *(self.max_comment_len   ), self.token_num) # with BOS  so +1
        # if len(self.code_modalities) > 1:
        #     self.fuse_linear = nn.Linear(self.hidden_size * len(self.code_modalities), self.hidden_size)



    #
    # def init_hidden(self, batch_size: int) -> Any:
    #     return self.rnn.init_hidden(batch_size)

    def forward(self, batch, tok_output, tok_enc_hc, sbtao_output ) -> Tuple:
        # sample_max, seq_length = sample_opt.get('sample_max', 1), \
        #                          sample_opt.get('seq_length', self.max_predict_length)
        # print('batch...')
        # pprint(batch)
        # (batch_size*mLen, batch_size*(mLen+1),)
        seq_length = self.max_comment_len
        comment, comment_input, comment_target, comment_len, raw_comment = batch['comment']
        device = comment.device
        batch_size = comment.size(0)
        seq_length = min(comment.size(1) + 1, seq_length)  # +1 is for EOS
        # input = torch.zeros(batch_size, 1).long().fill_(BOS).to(device)  # (batch_size*1) and all items are 2 (BOS)
        # Values that indicate whether [STOP] token has already been encountered; 1 => Not encountered, 0 otherwise
        # mask = torch.LongTensor(batch_size).fill_(1).to(device)  # (batch_size) and all items are 1
        seq, seq_logp_gathered = torch.zeros(batch_size, seq_length).long().to(device), \
                                                torch.zeros(batch_size, seq_length).to(device)



        seq_logprobs = torch.zeros(batch_size, seq_length, self.token_num).to(device)

        # seq_padding_mask, dec_output = [], []

        # dec_hidden = tok_enc_hc

        tok_output = tok_output.reshape(batch_size,-1,self.hidden_size)
        sbtao_output = sbtao_output.reshape(batch_size,-1,self.hidden_size)

        for t in range(seq_length):

            input =  comment_input[:, t,:] #bs,max_len

            input_emb = self.wemb(input)  # (batch_size,max_len , emb_size)
            dec_output, _ = self.rnn(input_emb,  tok_enc_hc)

            attn = torch.bmm(dec_output, tok_output.reshape(batch_size,tok_output.shape[-1],-1) )
            attn = torch.exp(F.log_softmax(attn, dim=-1))

            ast_attn = torch.bmm(dec_output , sbtao_output.reshape(batch_size,sbtao_output.shape[-1],-1))
            ast_attn = torch.exp(F.log_softmax(ast_attn, dim=-1))

                     # bs,max_len,tok_len            bs,tok_len,dim => bs,max_len,dim
            context = torch.bmm(attn, tok_output)
            ast_context = torch.bmm(ast_attn , sbtao_output)


            context = torch.cat([context, dec_output, ast_context],-1) # bs,max_len,dim*3
            out = torch.nn.functional.relu(self.linear_1(context)) # bs,max_len,dim
            # LOGGER.info("t/seq_length {}/{} after linear1 out.shape: {}".format(t,seq_length, out.shape ))
            out = torch.nn.functional.relu(self.linear_2(out.reshape(batch_size,-1)))# bs,max_len,dim => bs,comment_voca_dim

            logprobs = F.log_softmax(out, dim=-1)  # (batch_size*comment_dict_size)
            # prob_prev = torch.exp(logprobs)  # (batch_size*comment_dict_size)

            # if sample_max:
            sample_logprobs, predicted = torch.max(logprobs, 1)
            seq[:, t] = predicted.reshape(-1)
            seq_logp_gathered[:, t] = sample_logprobs
            seq_logprobs[:, t, :] = logprobs


        return seq, seq_logprobs



    def sample(self, batch, tok_output, tok_enc_hc, sbtao_output) -> Tuple:
        seq_length = self.max_comment_len
        comment, comment_input, comment_target, comment_len, raw_comment = batch['comment']
        device = comment.device
        batch_size = comment.size(0)
        seq_length = min(comment.size(1) + 1, seq_length)  # +1 is for EOS

        seq, seq_logp_gathered = torch.zeros(batch_size, seq_length).long().to(device), \
                                 torch.zeros(batch_size, seq_length).to(device)

        seq_logprobs = torch.zeros(batch_size, seq_length, self.token_num).to(device)


        tok_output = tok_output.reshape(batch_size, -1, self.hidden_size)
        sbtao_output = sbtao_output.reshape(batch_size, -1, self.hidden_size)

        input = torch.zeros(batch_size, seq_length ).long().to(device)
        input[:,0]= input[:,0].fill_(BOS).to(device)
        LOGGER.info("input.shape: {}".format(input.shape))
        for t in range(seq_length):
            # input = comment_input[:, t, :]  # bs,max_len
            input_emb = self.wemb(input)  # (batch_size,max_len , emb_size)
            dec_output, _ = self.rnn(input_emb, tok_enc_hc)

            attn = torch.bmm(dec_output, tok_output.reshape(batch_size, tok_output.shape[-1], -1))
            attn = torch.exp(F.log_softmax(attn, dim=-1))

            ast_attn = torch.bmm(dec_output, sbtao_output.reshape(batch_size, sbtao_output.shape[-1], -1))
            ast_attn = torch.exp(F.log_softmax(ast_attn, dim=-1))

            # bs,max_len,tok_len            bs,tok_len,dim => bs,max_len,dim
            context = torch.bmm(attn, tok_output)
            ast_context = torch.bmm(ast_attn, sbtao_output)

            context = torch.cat([context, dec_output, ast_context], -1)  # bs,max_len,dim*3
            out = torch.nn.functional.relu(self.linear_1(context))  # bs,max_len,dim
            out = torch.nn.functional.relu(
                self.linear_2(out.reshape(batch_size, -1)))  # bs,max_len,dim => bs,comment_voca_dim

            logprobs = F.log_softmax(out, dim=-1)  # (batch_size*comment_dict_size)
            # prob_prev = torch.exp(logprobs)  # (batch_size*comment_dict_size)

            # if sample_max:
            sample_logprobs, predicted = torch.max(logprobs, 1)
            seq[:, t] = predicted.reshape(-1)
            seq_logp_gathered[:, t] = sample_logprobs
            seq_logprobs[:, t, :] = logprobs
            input[:,t] = seq[:, t]

        return seq, seq_logprobs

class AstAttendGruV3Decoder(nn.Module):
    def __init__(self, config,token_num) -> None:
        super(AstAttendGruV3Decoder, self).__init__()
        # embedding params
        self.config = config
        self.token_num = token_num
        # self.embed_size = 100
        self.embed_size = config['training']['tok_embed_size']
        # rnn params
        # self.hidden_size = 256
        self.hidden_size = config['training']['rnn_hidden_size']
        self.layer_num = 1
        self.dropout =  0
        self.rnn_type = 'GRU'
        self.bidirectional = False
        # self.max_comment_len =  config['dataset']['max_comment_len']
        self.max_predict_length = self.config['training']['max_predict_length']


        # self.wemb = Encoder_Emb(self.token_num, self.embed_size, )
        # self.rnn = Encoder_RNN(self.rnn_type, self.embed_size, self.hidden_size, self.layer_num,
        #                        self.dropout, self.bidirectional)


        # self.wemb = torch.nn.Embedding(num_embeddings=self.token_num ,
        #                                   embedding_dim=self.embed_size,
        #                                padding_idx=self.config['training']['padding_idx'],
        #                                  max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False)
        #
        # self.rnn = torch.nn.GRU(  input_size=100, hidden_size=self.hidden_size ,  num_layers=self.layer_num,
        #                           bias= True,  batch_first=True,
        #                           dropout=self.dropout,  bidirectional=self.bidirectional)

        self.wemb = Encoder_Emb(self.token_num, self.embed_size, )

        self.rnn = Encoder_RNN(self.rnn_type, self.embed_size, self.hidden_size, self.layer_num,
                               dropout=self.dropout, bidirectional=self.bidirectional )

        # self.concat_map = nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size)
        self.linear_1 = nn.Linear(self.hidden_size*3, self.hidden_size)
        self.linear_2 = nn.Linear(self.hidden_size  , self.token_num) # with BOS  so +1
        # if len(self.code_modalities) > 1:
        #     self.fuse_linear = nn.Linear(self.hidden_size * len(self.code_modalities), self.hidden_size)



    #
    # def init_hidden(self, batch_size: int) -> Any:
    #     return self.rnn.init_hidden(batch_size)

    def forward(self, batch, tok_output, tok_enc_hc, sbtao_output ) -> Tuple:
        # sample_max, seq_length = sample_opt.get('sample_max', 1), \
        #                          sample_opt.get('seq_length', self.max_predict_length)
        # print('batch...')
        # pprint(batch)
        # (batch_size*mLen, batch_size*(mLen+1),)
        seq_length = self.max_predict_length
        comment, comment_input, comment_target, comment_len, raw_comment = batch['comment']
        device = comment.device
        batch_size = comment.size(0)
        seq_length = min(comment.size(1) + 1, seq_length)  # +1 is for EOS
        # input = torch.zeros(batch_size, 1).long().fill_(BOS).to(device)  # (batch_size*1) and all items are 2 (BOS)
        # Values that indicate whether [STOP] token has already been encountered; 1 => Not encountered, 0 otherwise
        # mask = torch.LongTensor(batch_size).fill_(1).to(device)  # (batch_size) and all items are 1
        seq, seq_logp_gathered = torch.zeros(batch_size, seq_length).long().to(device), \
                                                torch.zeros(batch_size, seq_length).to(device)



        seq_logprobs = torch.zeros(batch_size, seq_length, self.token_num).to(device)

        # seq_padding_mask, dec_output = [], []

        # dec_hidden = tok_enc_hc

        tok_output = tok_output.reshape(batch_size,-1,self.hidden_size)
        sbtao_output = sbtao_output.reshape(batch_size,-1,self.hidden_size)

        for t in range(seq_length):

            input =  comment_input[:, t].unsqueeze(1) #bs,max_len

            input_emb = self.wemb(input)  # (batch_size,max_len , emb_size)
            # dec_output, _ = self.rnn(input_emb,  tok_enc_hc)
            dec_output, _ = self.rnn(input_emb, hidden=tok_enc_hc)  # (batch_size*1*rnn_hidden_size, )

            attn = torch.bmm(dec_output, tok_output.reshape(batch_size,tok_output.shape[-1],-1) )
            attn = torch.exp(F.log_softmax(attn, dim=-1))

                            # bs,1,dim   bs,dim,sbtao_len => bs,1,sbtao_len
            ast_attn = torch.bmm(dec_output , sbtao_output.reshape(batch_size,sbtao_output.shape[-1],-1))
            ast_attn = torch.exp(F.log_softmax(ast_attn, dim=-1))

                     # bs,max_len,tok_len            bs,tok_len,dim => bs,max_len,dim
            context = torch.bmm(attn, tok_output)
            ast_context = torch.bmm(ast_attn , sbtao_output) # bs,1,sbtao_len bs,sbtao_len,dim => bs,1,dim


            context = torch.cat([context , dec_output , ast_context],-1).reshape(batch_size,-1)
            out = torch.nn.functional.relu(self.linear_1(context)) # bs,max_len,dim
            # LOGGER.info("t/seq_length {}/{} after linear1 out.shape: {}".format(t,seq_length, out.shape ))
            out = torch.nn.functional.relu(self.linear_2(out.reshape(batch_size,-1)))# bs,max_len,dim => bs,comment_voca_dim

            logprobs = F.log_softmax(out, dim=-1)  # (batch_size*comment_dict_size)
            # prob_prev = torch.exp(logprobs)  # (batch_size*comment_dict_size)

            # if sample_max:
            sample_logprobs, predicted = torch.max(logprobs, 1)
            seq[:, t] = predicted.reshape(-1)
            seq_logp_gathered[:, t] = sample_logprobs
            seq_logprobs[:, t, :] = logprobs


        return seq, seq_logprobs



    def sample(self, batch, tok_output, tok_enc_hc, sbtao_output) -> Tuple:
        seq_length = self.max_predict_length
        comment, comment_input, comment_target, comment_len, raw_comment = batch['comment']
        device = comment.device
        batch_size = comment.size(0)
        # seq_length = min(comment.size(1) + 1, seq_length)  # +1 is for EOS

        seq, seq_logp_gathered = torch.zeros(batch_size, seq_length).long().to(device), \
                                 torch.zeros(batch_size, seq_length).to(device)

        seq_logprobs = torch.zeros(batch_size, seq_length, self.token_num).to(device)


        tok_output = tok_output.reshape(batch_size, -1, self.hidden_size)
        sbtao_output = sbtao_output.reshape(batch_size, -1, self.hidden_size)

        # input = torch.zeros(batch_size, seq_length ).long().to(device)
        # input[:,0]= input[:,0].fill_(BOS).to(device)

        input = torch.zeros(batch_size, 1).long().fill_(BOS).to(device)

        # LOGGER.info("input.shape: {}".format(input.shape))
        for t in range(seq_length):
            # input = comment_input[:, t, :]  # bs,max_len
            input_emb = self.wemb(input)  # (batch_size,max_len , emb_size)
            # dec_output, _ = self.rnn(input_emb, tok_enc_hc)
            dec_output, _ = self.rnn(input_emb, hidden = tok_enc_hc)

            attn = torch.bmm(dec_output, tok_output.reshape(batch_size, tok_output.shape[-1], -1))
            attn = torch.exp(F.log_softmax(attn, dim=-1))

            ast_attn = torch.bmm(dec_output, sbtao_output.reshape(batch_size, sbtao_output.shape[-1], -1))
            ast_attn = torch.exp(F.log_softmax(ast_attn, dim=-1))

            # bs,max_len,tok_len            bs,tok_len,dim => bs,max_len,dim
            context = torch.bmm(attn, tok_output)
            ast_context = torch.bmm(ast_attn, sbtao_output)

            context = torch.cat([context, dec_output, ast_context], -1).reshape(batch_size,-1)  # bs,max_len,dim*3
            out = torch.nn.functional.relu(self.linear_1(context))  # bs,max_len,dim
            out = torch.nn.functional.relu(
                self.linear_2(out.reshape(batch_size, -1)))  # bs,max_len,dim => bs,comment_voca_dim

            logprobs = F.log_softmax(out, dim=-1)  # (batch_size*comment_dict_size)
            # prob_prev = torch.exp(logprobs)  # (batch_size*comment_dict_size)

            # if sample_max:
            sample_logprobs, predicted = torch.max(logprobs, 1)
            seq[:, t] = predicted.reshape(-1)
            seq_logp_gathered[:, t] = sample_logprobs
            seq_logprobs[:, t, :] = logprobs
            # input[:,t] = seq[:, t]
            input = predicted.reshape(-1, 1)

        return seq, seq_logprobs


