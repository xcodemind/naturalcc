# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from ncc import LOGGER
from ncc.model.template import *
from ncc.module.code2vec.multi_modal import *
from ncc.module.code2vec.base import *
from typing import Dict, Any


class MMRewardModel(CodeEnc_CmntEnc):

    def __init__(self, config: Dict, ) -> None:
        LOGGER.debug('building {}...'.format(self.__class__.__name__))
        super(MMRewardModel, self).__init__(
            config=config,
            code_encoder=MMEncoder_EmbRNN.load_from_config(config),
            comment_encoder=Encoder_Conv2d.load_from_config(config),
        )
        self.config = config

        # self.out_dim = len(config['training']['conv2d_kernels']) * config['training']['conv2d_out_channels'] + config['training']['embed_size']

        # self.value = nn.Linear(config['rnn_hidden_size'], 1)
        self.proj = nn.Linear(config['training']['rnn_hidden_size'], config['training']['rnn_hidden_size'])
        self.fc = nn.Linear(len(config['training']['conv2d_kernels']) * config['training']['conv2d_out_channels'] + config['training']['rnn_hidden_size'], 1, bias=True)
        self.dropout = config['arel']['reward_model_dropout']

        if config['arel']['reward_model_activation'] == 'linear':
            self.activation = None
        elif config['arel']['reward_model_activation'] == 'sign':
            self.activation = nn.Softsign()
        elif config['arel']['reward_model_activation'] == 'tahn':
            self.activation = nn.Tanh()

    def forward(self, batch, comment, ) -> Any:
        # _, comment_logprobs, _, _, _, = self.train_pipeline(batch)
        # if self.config['training']['pointer']:
        #     code_dict_comment, comment_extend_vocab, pointer_extra_zeros, code_oovs = batch['pointer']
        # else:
        #     code_oovs = None
        # enc_output, dec_hidden, enc_mask = model.encoder.forward(batch)
        # sample_opt = {'sample_max': 0, 'seq_length': self.config['max_predict_length']}
        # comment, comment_logprobs, comment_logp_gathered, comment_padding_mask, reward, comment_lprob_sum, \
        # dec_output, dec_hidden, = model.decoder.forward_pg(batch, enc_output, dec_hidden, enc_mask, token_dicts,
        #                                                    sample_opt, reward_func, code_oovs)
        # critic
        enc_output, dec_hidden, enc_mask = self.code_encoder.forward(batch)
        # sample_opt = {'sample_max': 0, 'seq_length': self.config['training']['max_predict_length']}
        # comment, comment_logprobs, comment_logp_gathered, comment_padding_mask, comment_lprob_sum, dec_hidden,
        # comment_critic, comment_logprobs_critic, comment_logp_gathered_critic, comment_padding_mask_critic, comment_lprob_sum, \
        # dec_output_critic, dec_hidden_critic, = self.decoder.forward(batch, enc_output_critic, dec_hidden_critic,
        #                                                              enc_mask_critic, sample_opt)
        comment_emb = self.comment_encoder.forward(comment)

        # value = self.value(dec_output_critic.reshape(-1, dec_output_critic.size(-1))).view_as(reward) # (batch_size*comment_len)
        print('dec_hidden[0]: ', dec_hidden[0].size())
        print('dec_hidden[0][-1]: ', dec_hidden[0][-1].size())
        code_emb = self.proj(dec_hidden[0][-1])
        print('code_emb: ', code_emb.size())
        print('comment_emb: ', comment_emb.size())
        combined = torch.cat([code_emb, comment_emb], 1)
        print('combined: ', combined.size())
        # self.activation
        prob = self.fc(F.dropout(combined, self.dropout, training=self.training)).view(-1)
        print('prob: ', prob.size())
        reward = self.activation(prob)
        print('reward: ', reward.size())
        return reward


class MMRewardModel_(nn.Module):
    def __init__(self, opt):
        super(MMRewardModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.word_embed_dim = 300
        self.feat_size = opt.feat_size
        self.kernel_num = 512
        self.kernels = [2, 3, 4, 5]
        self.out_dim = len(self.kernels) * self.kernel_num + self.word_embed_dim

        self.emb = nn.Embedding(self.vocab_size, self.word_embed_dim)
        # self.emb.weight.data.copy_(torch.from_numpy(np.load("VIST/embedding.npy")))

        self.proj = nn.Linear(self.feat_size, self.word_embed_dim)

        self.convs = [nn.Conv2d(1, self.kernel_num, (k, self.word_embed_dim)) for k in self.kernels]

        self.dropout = nn.Dropout(opt.dropout)
        print('out-dim: ', self.out_dim)
        self.fc = nn.Linear(self.out_dim, 1, bias=True)

        if opt.activation.lower() == "linear":
            self.activation = None
        elif opt.activation.lower() == "sign":
            self.activation = nn.Softsign()
        elif self.activation.lower() == "tahn":
            self.activation = nn.Tanh()

    def forward(self, story, feature):
        # embedding = Variable(self.emb(story).data)  # (batch_size, seq_length, embed_dim)
        embedding = self.emb(story) # (batch_size, seq_length, embed_dim)

        self.convs = [model.cuda() for model in self.convs]

        # batch x seq_len x emb_dim -> batch x 1 x seq_len x emb_dim
        embedding = embedding.unsqueeze(1)
        x = [F.relu(conv(embedding)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        print('reward-x: ')
        print(x)
        # combine with image feature
        img = self.proj(feature)
        print('reward-img: ')
        print(img)
        combined = torch.cat([x, img], 1)
        print('reward-combined: ')
        print(combined)
        combined = self.dropout(combined)

        prob = self.fc(combined).view(-1)

        return self.activation(prob)