# -*- coding: utf-8 -*-

import sys

sys.path.append('.')
import torch
import torch.nn as nn
import torch.nn.functional as F
# from src.utils.util import to_cuda,clean_up_sentence
from src.utils.util import masked_softmax

from  src.utils.constants import *

class DotSimiAttn(nn.Module):
    def __init__(self, rnn_hidden_size):
        super(DotSimiAttn, self).__init__()
        # self.opt = opt
        self.rnn_hidden_size = rnn_hidden_size
        
    
    # def forward(self, st_hat, h, enc_padding_mask, sum_temporal_srcs):
    def forward(self,  h,enc_out, enc_padding_mask,batchsize):

        # et = self.W_h(h)  # bs,n_seq, n_hid
        # dec_fea = self.W_s(st_hat).unsqueeze(1)  # bs,1,2*n_hid
        # et = et + dec_fea
        # et = torch.tanh(et)  # bs,n_seq, n_hid
        # et = self.v(et).squeeze(2)  # bs,n_seq

        # bs,seq,dim  bs,dim,1    => bs,seq,1
        enc_out = enc_out.reshape(batchsize,-1,self.rnn_hidden_size)
        et = torch.bmm(enc_out  , h.reshape(batchsize,self.rnn_hidden_size,1)).reshape(batchsize,-1)

        at = masked_softmax(vector=et,
                   mask=enc_padding_mask,
                   dim = 1,
                   memory_efficient = False
                   )

        at = at.unsqueeze(1)  # bs,1,n_seq
        # Compute encoder context vector
        ct_e = torch.bmm(at,enc_out)  #  bs,1,n_seq  x  bs,n_seq,n_hid =   bs, 1,  n_hid
        ct_e = ct_e.squeeze(1) # bs,  n_hid

        # return ct_e, at, sum_temporal_srcs
        return ct_e

class DecoderStep(nn.Module):
    def __init__(self,embed_size,rnn_hidden_size, dict_comment,dropout):
        super(DecoderStep, self).__init__()
        self.rnn_hidden_size = rnn_hidden_size
        self.embed_size =  embed_size
        self.W1 = nn.Linear(self.rnn_hidden_size , self.rnn_hidden_size,bias=False  )
        self.W2 = nn.Linear(self.rnn_hidden_size , self.rnn_hidden_size,bias=False  )
        self.W = nn.Linear(self.rnn_hidden_size, dict_comment.size )
        self.wemb = nn.Embedding(dict_comment.size, self.embed_size, padding_idx= PAD)
        self.lstm = nn.LSTMCell(self.embed_size, self.rnn_hidden_size)
        self.enc_attention = DotSimiAttn(rnn_hidden_size)
        self.dropout = dropout

    def forward(self, x_t, s_t, enc_out, enc_padding_mask):

        # print("=========== x_t.shape: ",x_t.shape )
        batchsize = x_t.shape[0]
        x = self.wemb(x_t)
        # print("=== x_t.shape: ", x_t.shape)
        # if s_t is not None:
            # print("=== s_t[0].shape: ", s_t[0].shape)
            # print("=== s_t[1].shape: ", s_t[1].shape)
        # tmp_h, tmp_c = s_t
        # tmp_h = tmp_h.reshape(x.size()[0], -1)
        # tmp_c = tmp_c.reshape(x.size()[0], -1)
        # s_t = (tmp_h, tmp_c)

        s_t = self.lstm(x, s_t)  # 因为用的 LSTMCell ,所以s_t 只是 (h,c) ,而不是nn.LSTM得到的output, (h_n, c_n)

        # tmp_h, tmp_c = s_t
        # tmp_h = tmp_h.reshape(x.size()[0], -1)
        # tmp_c = tmp_c.reshape(x.size()[0], -1)
        # s_t = (tmp_h, tmp_c)

        dec_h, dec_c = s_t
        # ct_e  = self.enc_attention(dec_h, enc_out, enc_padding_mask)
        ct_e  = self.enc_attention(dec_h, enc_out, enc_padding_mask, batchsize)

        # print("dec_h.shape: ",dec_h.shape)
        # print("ct_e.shape: ",ct_e.shape)
        out = self.W(torch.tanh(F.dropout(self.W1(dec_h), self.dropout, training=self.training)+
                                F.dropout(self.W2(ct_e), self.dropout, training=self.training)))  # bs, n_vocab
        # vocab_dist = F.softmax(out, dim=1)
        logprobs = F.log_softmax(out, dim=1)

        return logprobs ,s_t



class CodeNNSeqDecoder(nn.Module):
    # max_predict_length = config['training']['max_predict_length'],
    def __init__(self, max_predict_length,dict_comment):
        super(CodeNNSeqDecoder, self).__init__()
        embed_size = 400
        rnn_hidden_size = 400
        self.dict_comment = dict_comment
        self.dropout = 0.5
        # self.dec_attention = decoder_attention(opt)
        self.dec_step = DecoderStep(embed_size,rnn_hidden_size, dict_comment,self.dropout)
        self.max_predict_length = max_predict_length
        # if self.opt.init_type == "xulu":
            #  init_lstm_wt(self.lstm)
            # pass  # 那个代码用的双向lstm，init_lstm_wt里面对于bias的处理可能和我们的单向lstm不一样
        # self.V1 = nn.Linear(self.rnn_hidden_size, self.dict_comment.size())



    def forward(self,  comment_input_batch,  enc_out, enc_padding_mask, s_t=None , sample_opt={}):
        sample_max, seq_length = sample_opt.get('sample_max', 1), sample_opt.get('seq_length',
                                                                                 self.max_predict_length)

        seq_length = min(comment_input_batch.size(1), seq_length)
        batch_size = enc_out.shape[0]
        # x_t = torch.zeros(batch_size, 1).long().fill_( BOS)
        # x_t = to_cuda(self.opt, x_t)
        seq = torch.zeros(batch_size, seq_length).long().cuda()
        seq_logprobs = torch.zeros(batch_size, seq_length, self.dict_comment.size ).cuda()

        for t in range(seq_length):
            # x_t = comment_input_batch[:, t].unsqueeze(1) # teacher force
            x_t = comment_input_batch[:, t] # teacher force
            # print("t: ",t)
            # print("------  x_t.shape: ",x_t.shape )
            x_t = x_t.reshape(batch_size)
            logprobs ,s_t = self.dec_step( x_t, s_t, enc_out, enc_padding_mask)
            sample_logprobs, predicted = torch.max(logprobs, 1)
            seq[:, t] = predicted.reshape(-1)
            # seq_logp_gathered[:, t] = sample_logprobs
            seq_logprobs[:, t, :] = logprobs
            # x_t = predicted.reshape(-1, 1)

        return seq, seq_logprobs


    def sample(self,  comment_input_batch,  enc_out, enc_padding_mask, s_t=None , sample_opt={}):
        sample_max, seq_length = sample_opt.get('sample_max', 1), sample_opt.get('seq_length',
                                                                                 self.max_predict_length)

        seq_length = min(comment_input_batch.size(1), seq_length)
        batch_size = enc_out.shape[0]
        # x_t = torch.zeros(batch_size, 1).long().fill_( BOS)
        x_t = torch.zeros(batch_size).long().fill_( BOS).cuda()
        # x_t = to_cuda(self.opt, x_t)
        seq = torch.zeros(batch_size, seq_length).long().cuda()
        seq_logprobs = torch.zeros(batch_size, seq_length, self.dict_comment.size ).cuda()

        for t in range(seq_length):
            logprobs ,s_t = self.dec_step( x_t, s_t, enc_out, enc_padding_mask)
            sample_logprobs, predicted = torch.max(logprobs, 1)
            seq[:, t] = predicted.reshape(-1)
            # seq_logp_gathered[:, t] = sample_logprobs
            seq_logprobs[:, t, :] = logprobs
            # x_t = predicted.reshape(-1, 1)
            x_t = predicted.reshape(-1)

        return seq, seq_logprobs





