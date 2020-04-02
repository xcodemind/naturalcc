# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from ncc import LOGGER
from ncc.model.template import CodeEnc_CmntEnc, IModel
from ncc.module.code2vec.multi_modal import MMEncoder_EmbRNN
from ncc.module.code2vec.base import Encoder_Conv2d
from ncc.metric import BaseLoss
from ncc.utils.constants import *
from typing import Dict, Any

class MMDiscriminator(CodeEnc_CmntEnc):

    def __init__(self, config: Dict, ) -> None:
        LOGGER.debug('building {}...'.format(self.__class__.__name__))
        super(MMDiscriminator, self).__init__(
            config=config,
            code_encoder=MMEncoder_EmbRNN.load_from_config(config),
            comment_encoder=Encoder_Conv2d.load_from_config(config),
        )
        self.config = config
        # self.value = nn.Linear(config['rnn_hidden_size'], 1)
        self.proj = nn.Linear(config['training']['rnn_hidden_size'], config['training']['rnn_hidden_size'])
        self.fc = nn.Linear(len(config['training']['conv2d_kernels']) * config['training']['conv2d_out_channels'] + config['training']['rnn_hidden_size'], 1, bias=True)
        if config['training']['activation'] == 'linear':
            self.activation = None
        elif config['training']['activation'] == 'sign':
            self.activation = nn.Softsign()
        elif config['training']['activation'] == 'tahn':
            self.activation = nn.Tanh()

    def train_sl(self, model: IModel, batch: Dict, criterion: BaseLoss, label: torch.Tensor, ) -> Any:
        # _, comment_logprobs, _, _, _, = self.train_pipeline(batch)
        if self.config['training']['pointer']:
            code_dict_comment, comment_extend_vocab, pointer_extra_zeros, code_oovs = batch['pointer']
        else:
            code_oovs = None
        # enc_output, dec_hidden, enc_mask = model.encoder.forward(batch)
        # sample_opt = {'sample_max': 0, 'seq_length': self.config['max_predict_length']}
        # comment, comment_logprobs, comment_logp_gathered, comment_padding_mask, reward, comment_lprob_sum, \
        # dec_output, dec_hidden, = model.decoder.forward_pg(batch, enc_output, dec_hidden, enc_mask, token_dicts,
        #                                                    sample_opt, reward_func, code_oovs)
        # critic
        enc_output_critic, dec_hidden_critic, enc_mask_critic = self.code_encoder.forward(batch)
        sample_opt = {'sample_max': 0, 'seq_length': self.config['training']['max_predict_length']}
        # comment, comment_logprobs, comment_logp_gathered, comment_padding_mask, comment_lprob_sum, dec_hidden,
        # comment_critic, comment_logprobs_critic, comment_logp_gathered_critic, comment_padding_mask_critic, comment_lprob_sum, \
        # dec_output_critic, dec_hidden_critic, = self.decoder.forward(batch, enc_output_critic, dec_hidden_critic,
        #                                                              enc_mask_critic, sample_opt)
        xxx = self.comment.forward()

        # value = self.value(dec_output_critic.reshape(-1, dec_output_critic.size(-1))).view_as(reward) # (batch_size*comment_len)
        # self.proj
        # self.fc
        # self.activation
        disc_loss = criterion(value, label) # value: (batch_size*comment_len), reward: (batch_size*comment_len)
        # disc_loss = critic_loss * comment_padding_mask
        # critic_loss = torch.sum(critic_loss) / torch.sum(comment_padding_mask).float()  # comment.data.ne(data.Constants.PAD)
        # print('critic_loss: ', critic_loss.item())
        # assert False
        return disc_loss




# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# import numpy as np
# import ncc.data.codesum.constants as constants
# from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class Discriminator(nn.Module):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    Highway architecture based on the pooled feature maps is added. Dropout is adopted.
    """

    def __init__(self, wemb, filter_sizes, num_filters, opt):  # code_encoder
        super(Discriminator, self).__init__()
        self.opt = opt
        # self.code_encoder = code_encoder
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_f, (f_size, opt.ninp)) for f_size, num_f in zip(filter_sizes, num_filters)
        ])
        self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        # self.dropout = nn.Dropout(p=dropout_prob)
        self.linear = nn.Linear(sum(num_filters), 2)
        self.wemb = wemb
        # self.wemb = wemb
        # self.embed = nn.Embedding(vocab_size, embedding_dim)
        # self.convs = nn.ModuleList([
        #     nn.Conv2d(1, num_f, (f_size, embedding_dim)) for f_size, num_f in zip(filter_sizes, num_filters)
        # ])
        # self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        # self.dropout = nn.Dropout(p = dropout_prob)
        # self.fc = nn.Linear(sum(num_filters), num_classes)

    def forward(self, comment):  # code, code_length,
        """
        Inputs: x
            - x: (batch_size, seq_len)
        Outputs: out
            - out: (batch_size, num_classes)
        """
        # print('Discriminator-comment: ', comment.type(), comment.size())
        # print(comment)
        emb = self.wemb(comment.t()).unsqueeze(1)  # batch_size, 1 * seq_len * emb_dim
        # print('Discriminator-emb: ', emb.type(), emb.size())
        # print(emb)

        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * seq_len]
        # print('Discriminator-convs: ')
        # print(convs)
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]  # [batch_size * num_filter]
        # print('Discriminator-pools: ')
        # print(pools)
        out = torch.cat(pools, 1)  # batch_size * sum(num_filters)
        # print('Discriminator-out: ', out.type(), out.size())
        # print(out)
        highway = self.highway(out)
        # print('Discriminator-highway: ', highway.type(), highway.size())
        # print(highway)
        transform = torch.sigmoid(highway)
        # print('Discriminator-transform: ', transform.type(), transform.size())
        # print(transform)
        out = transform * F.relu(highway) + (1. - transform) * out  # sets C = 1 - torch
        # print('Discriminator-out-: ', out.type(), out.size())
        # print(out)
        encoder_feat = self.linear(F.dropout(out, self.opt.dropout, training=self.training))
        # print('Discriminator-encoder_feat: ', encoder_feat.type(), encoder_feat.size())
        # print(encoder_feat)
        out = torch.log_softmax(encoder_feat, dim=1)  # batch * num_classes
        # print('Discriminator-out--: ', out.type(), out.size())
        # print(out)

        return out


class Discriminator_LSTM_(nn.Module):
    def __init__(self, opt, dict_comment):
        super(Discriminator_LSTM, self).__init__()

        self.opt = opt
        self.linear = nn.Linear(opt.nhid, 2)
        self.wemb = nn.Embedding(dict_comment.size(), opt.ninp, padding_idx=PAD)
        self.rnn = getattr(nn, opt.decoder_rnn_type)(opt.ninp, opt.nhid, opt.nlayers, dropout=opt.dropout)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data

        # if self.rnn_type == 'LSTM':
        return (weight.new(self.opt.nlayers, bsz, self.opt.nhid).zero_().requires_grad_(),
                weight.new(self.opt.nlayers, bsz, self.opt.nhid).zero_().requires_grad_())

    def forward(self, comment, comment_length):
        # print('discriminator-comment: ', comment.type(), comment.size())
        # print(comment)
        # print('discriminator-comment_length: ')
        # print(comment_length)
        hidden = self.init_hidden(comment.size(1))  # self.opt.batch_size*2
        # print('discriminator-hidden[0]: ', hidden[0].type(), hidden[0].size())
        # print(hidden[0])
        emb = self.wemb(comment)
        # print('discriminator-emb: ', emb.type(), emb.size())
        # print(emb)
        emb_packed = pack_padded_sequence(emb, comment_length)
        # print('discriminator-emb_packed: ')
        # print(emb_packed)
        output, hidden = self.rnn(emb_packed, hidden)
        # print('discriminator-output: ')
        # print(output)
        # print('discriminator-hidden[0]-: ', hidden[0].type(), hidden[0].size())
        # print(hidden[0])
        encoder_feat = self.linear(F.dropout(hidden[0], self.opt.dropout, training=self.training)).squeeze()
        # print('discriminator-encoder_feat: ', encoder_feat.type(), encoder_feat.size())
        # print(encoder_feat)
        logprob = torch.log_softmax(encoder_feat, dim=1)
        # print('discriminator-logprob: ', logprob.type(), logprob.size())
        # print(logprob)

        return logprob


class Discriminator_LSTM(nn.Module):
    def __init__(self, opt, dict_comment):
        super(Discriminator_LSTM, self).__init__()

        self.opt = opt
        self.linear = nn.Linear(opt.nhid, 2)
        self.wemb = nn.Embedding(dict_comment.size(), opt.ninp, padding_idx=PAD)
        self.rnn = getattr(nn, opt.decoder_rnn_type)(opt.ninp, opt.nhid, opt.nlayers, dropout=opt.dropout)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data

        # if self.rnn_type == 'LSTM':
        return (weight.new(self.opt.nlayers, bsz, self.opt.nhid).zero_().requires_grad_(),
                weight.new(self.opt.nlayers, bsz, self.opt.nhid).zero_().requires_grad_())

    def forward(self, comment):
        # print('discriminator-comment: ', comment.type(), comment.size())
        # print(comment)
        # print('discriminator-comment_length: ')
        # print(comment_length)
        hidden = self.init_hidden(comment.size(1))  # self.opt.batch_size*2
        # print('discriminator-hidden[0]: ', hidden[0].type(), hidden[0].size())
        # print(hidden[0])
        emb = self.wemb(comment)
        # print('discriminator-emb: ', emb.type(), emb.size())
        # print(emb)
        # emb_packed = pack_padded_sequence(emb, comment_length)
        # print('discriminator-emb_packed: ')
        # print(emb_packed)
        output, hidden = self.rnn(emb, hidden)
        # print('discriminator-output: ')
        # print(output)
        # print('discriminator-hidden[0]-: ', hidden[0].type(), hidden[0].size())
        # print(hidden[0])
        encoder_feat = self.linear(F.dropout(hidden[0], self.opt.dropout, training=self.training)).squeeze()
        # print('discriminator-encoder_feat: ', encoder_feat.type(), encoder_feat.size())
        # print(encoder_feat)
        logprob = torch.log_softmax(encoder_feat, dim=1)
        # print('discriminator-logprob: ', logprob.type(), logprob.size())
        # print(logprob)

        return logprob