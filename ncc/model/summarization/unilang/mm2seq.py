# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

from ncc import *
from ncc.eval import *
from ncc.model.template import *
from ncc.module.code2vec.multi_modal import *
from ncc.module.summarization import *
from ncc.model import *
from ncc.dataset import *
from ncc.metric import *
from ncc.utils.util_data import batch_to_cuda
from ncc.data import *


class MM2Seq(Encoder2Decoder):

    def __init__(self, config: Dict) -> None:
        LOGGER.debug('building {}...'.format(self.__class__.__name__))
        super(MM2Seq, self).__init__(
            encoder=MMEncoder_EmbRNN.load_from_config(config),
            decoder=SeqDecoder.load_from_config(config, modal='comment'),
        )
        self.config = config

    def eval_pipeline(self, batch_data: Dict, ) -> Tuple:
        # train/eval pipeline may be quite different, therefore we design two methods
        enc_output, dec_hidden, enc_mask = self.encoder.forward(batch_data)
        sample_opt = {'beam_size': 1, 'sample_max': 1, 'seq_length': self.config['training']['max_predict_length']}
        comment_pred, comment_logprobs, _, _ = \
            self.decoder.sample(batch_data, enc_output, dec_hidden, enc_mask, sample_opt)
        # print('comment_target_padded: ', comment_target_padded.size())

        return comment_pred, comment_logprobs  # , comment_target_padded,



    def train_sl(self, batch: Dict, criterion: BaseLoss, ) -> Any:
        # _, comment_logprobs, _, _, _, = self.train_pipeline(batch)
        enc_output, dec_hidden, enc_mask = self.encoder.forward(batch)
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

    def train_pg(self, batch: Dict, criterion: BaseLoss, token_dicts: TokenDicts, reward_func: str) -> Any:
        if self.config['training']['pointer']:
            code_dict_comment, comment_extend_vocab, pointer_extra_zeros, code_oovs = batch['pointer']
        else:
            code_oovs = None
        # enc_output: {'tok': batch_size*code_len*rnn_hidden_size}
        # dec_hidden: (batch_size*1*rnn_hidden_size, batch_size*1*rnn_hidden_size)
        # enc_mask: {'tok': batch_size*code_len}
        enc_output, dec_hidden, enc_mask = self.encoder.forward(batch)  #
        sample_opt = {'sample_max': 0, 'seq_length': self.config['training']['max_predict_length']}
        # comment: batch_size*comment_len
        # comment_logprobs: batch_size*comment_len*comment_dict_size
        # comment_logp_gathered: batch_size*comment_len
        # comment_padding_mask: batch_size*comment_len
        # reward: batch_size*comment_len
        # comment_lprob_sum: batch_size*comment_len
        # _: dec_hidden
        comment, comment_logprobs, comment_logp_gathered, comment_padding_mask, reward, comment_lprob_sum, _, _, \
            = self.decoder.forward_pg(batch, enc_output, dec_hidden, enc_mask, token_dicts, sample_opt, reward_func,
                                      code_oovs)
        # print('comment_logprobs: ', comment_logprobs.size())
        # print('comment: ', comment.size())
        rl_loss = criterion(comment_logprobs.reshape(-1, comment_logprobs.size(-1)), comment.reshape(-1, 1),
                            comment_padding_mask, reward)
        rl_loss = rl_loss / torch.sum(comment_padding_mask).float()  # comment.data.ne(data.Constants.PAD)
        # print('rl_loss: ', rl_loss.item())
        # assert False
        return rl_loss

    def train_sc(self, batch: Dict, criterion: BaseLoss, token_dicts: TokenDicts, reward_func: str, ) -> Any:
        if self.config['training']['pointer']:
            code_dict_comment, comment_extend_vocab, pointer_extra_zeros, code_oovs = batch['pointer']
        else:
            code_oovs = None
        # enc_output: {'tok': batch_size*code_len*rnn_hidden_size}
        # dec_hidden: (batch_size*1*rnn_hidden_size, batch_size*1*rnn_hidden_size)
        # enc_mask: {'tok': batch_size*code_len}
        enc_output, dec_hidden, enc_mask = self.encoder.forward(batch)
        sample_opt = {'sample_max': 0, 'seq_length': self.config['training']['max_predict_length']}
        # comment: batch_size*comment_len
        # comment_logprobs: batch_size*comment_len*comment_dict_size
        # comment_logp_gathered: batch_size*comment_len
        # comment_padding_mask: batch_size*comment_len
        # reward: batch_size*comment_len
        # comment_lprob_sum: batch_size*comment_len
        # _: dec_hidden
        comment, comment_logprobs, comment_logp_gathered, comment_padding_mask, reward, comment_lprob_sum, _, _, \
            = self.decoder.forward_pg(batch, enc_output, dec_hidden, enc_mask, token_dicts, sample_opt, reward_func,
                                      code_oovs)

        with torch.autograd.no_grad():
            sample_opt = {'sample_max': 1, 'seq_length': self.config['training']['max_predict_length']}
            comment2, comment_logprobs2, comment_logp2_gathered, comment_padding_mask2, reward2, comment_lprob_sum, _, _, = \
                self.decoder.forward_pg(batch, enc_output, dec_hidden, enc_mask, token_dicts, sample_opt, reward_func,
                                        code_oovs)  # 100x9,100x9
        # print('reward: ', reward.size())
        # print(reward)
        # print('reward2: ', reward2.size())
        # print(reward2)
        # print('reward - reward2: ', (reward - reward2).size())
        # print(reward - reward2)
        rl_loss = criterion(comment_logprobs, comment, comment_padding_mask, (reward - reward2))
        rl_loss = rl_loss / torch.sum(comment_padding_mask).float()  # comment.data.ne(data.Constants.PAD)
        # print('rl_loss: ', rl_loss.item())
        # assert False
        return rl_loss

    def train_sl_kd(self, batch: Dict, criterion: BaseLoss, ) -> Any:

        enc_output, dec_hidden, enc_mask = self.encoder.forward(batch)
        sample_opt = {'sample_max': 1, 'seq_length': self.config['training']['max_predict_length']}
        _, comment_logprobs, _, _, _, _, _, = self.decoder.forward(batch, enc_output, dec_hidden, enc_mask, sample_opt)
        # if self.opt.enc_hc2init_dec_mode == 'hinput':
        #     feat_final = torch.tanh(
        #         self.model.linear(F.dropout(dec_hidden[0], self.opt.dropout, training=self.model.training))). \
        #         reshape(-1, 1, self.opt.ninp)
        #     dec_hidden = self.model.decoder.init_hidden(feat_final.shape[0])
        #     _, dec_hidden = self.model.decoder.rnn(feat_final, dec_hidden)
        #
        # comment, comment_logprobs, comment_logp_gathered, comment_padding_mask, comment_lprob_sum = self.model.decoder.forward(
        #     batch, enc_output, dec_hidden, enc_padding_mask, sample_opt)

        if self.config['training']['pointer']:
            comment_target_batch2use = batch['pointer'][1]
        else:
            comment_target_batch2use = batch['comment'][2]

        if self.config['kd']['distill']:
            comment_loss = criterion(comment_logprobs, comment_target_batch2use, batch)
        else:
            comment_loss = criterion(comment_logprobs, comment_target_batch2use)

        return comment_loss, comment_logprobs, comment_target_batch2use

    def train_dtrl(self, batch: Dict, criterion: BaseLoss, token_dicts: TokenDicts, reward_func: str, ) -> Any:
        if self.config['training']['pointer']:
            code_dict_comment, comment_extend_vocab, pointer_extra_zeros, code_oovs = batch['pointer']
        else:
            code_oovs = None
        enc_output, dec_hidden, enc_mask = self.encoder.forward(batch)

        # calculate greedy reward
        sample_opt = {'sample_max': 1, 'seq_length': self.config['training']['max_predict_length']}
        _, comment_logprobs, _, comment_mask, greedy_reward, _, _, _, \
            = self.decoder.forward_pg(batch, enc_output, dec_hidden, enc_mask, token_dicts, sample_opt, reward_func,
                                      code_oovs)
        # calculate sample reward
        with torch.autograd.no_grad():
            sample_opt = {'sample_max': 0, 'seq_length': self.config['training']['max_predict_length']}
            _, _, _, _, critic_reward, _, _, _, \
                = self.decoder.forward_pg(batch, enc_output, dec_hidden, enc_mask, token_dicts, sample_opt, reward_func,
                                          code_oovs)

        if self.config['training']['pointer']:
            comment_target = batch['pointer'][1][:, :self.config['training']['max_predict_length']]
        else:
            comment_target = batch['comment'][2][:, :self.config['training']['max_predict_length']]

        reward_diff = (greedy_reward - critic_reward).clamp(1e-15)
        dtrl_loss = criterion(comment_logprobs, comment_target, reward_diff)
        return dtrl_loss
