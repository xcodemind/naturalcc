# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

from src import *
from src.eval import *
from src.model.template import *
from src.module.code2vec.multi_modal import *
from src.module.summarization import *
from src.model import *
from src.dataset import *
from src.metric import *
from src.utils.util_data import batch_to_cuda
from src.data import *


class MMCritic(Encoder2Decoder):

    def __init__(self, config: Dict) -> None:
        LOGGER.debug('building {}...'.format(self.__class__.__name__))
        super(MMCritic, self).__init__(
            encoder=MMEncoder_EmbRNN.load_from_config(config),
            decoder=SeqDecoder.load_from_config(config, modal='comment'),
        )
        self.config = config
        self.value = nn.Linear(config['training']['rnn_hidden_size'], 1)

    def train_sl(self, model: IModel, batch: Dict, criterion: BaseLoss, token_dicts: TokenDicts, reward_func: str, ) -> Any:
        # _, comment_logprobs, _, _, _, = self.train_pipeline(batch)
        if self.config['training']['pointer']:
            code_dict_comment, comment_extend_vocab, pointer_extra_zeros, code_oovs = batch['pointer']
        else:
            code_oovs = None
        enc_output, dec_hidden, enc_mask = model.encoder.forward(batch)
        sample_opt = {'sample_max': 0, 'seq_length': self.config['training']['max_predict_length']}
        comment, comment_logprobs, comment_logp_gathered, comment_padding_mask, reward, comment_lprob_sum, \
        dec_output, dec_hidden, = model.decoder.forward_pg(batch, enc_output, dec_hidden, enc_mask, token_dicts,
                                                           sample_opt, reward_func, code_oovs)
        # critic
        enc_output_critic, dec_hidden_critic, enc_mask_critic = self.encoder.forward(batch)
        sample_opt = {'sample_max': 0, 'seq_length': self.config['training']['max_predict_length']}
        # comment, comment_logprobs, comment_logp_gathered, comment_padding_mask, comment_lprob_sum, dec_hidden,
        comment_critic, comment_logprobs_critic, comment_logp_gathered_critic, comment_padding_mask_critic, comment_lprob_sum, \
        dec_output_critic, dec_hidden_critic, = self.decoder.forward(batch, enc_output_critic, dec_hidden_critic,
                                                                     enc_mask_critic, sample_opt)
        value = self.value(dec_output_critic.reshape(-1, dec_output_critic.size(-1))).view_as(reward) # (batch_size*comment_len)
        critic_loss = criterion(value, reward) # value: (batch_size*comment_len), reward: (batch_size*comment_len)
        critic_loss = critic_loss * comment_padding_mask
        critic_loss = torch.sum(critic_loss) / torch.sum(comment_padding_mask).float()  # comment.data.ne(data.Constants.PAD)
        # print('critic_loss: ', critic_loss.item())
        # assert False
        return critic_loss