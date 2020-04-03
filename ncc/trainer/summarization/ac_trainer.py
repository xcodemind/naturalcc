# -*- coding: utf-8 -*-
import os
import datetime
import time
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from ncc import LOGGER
from ncc.trainer.trainer_ import Trainer
from ncc.model.template import IModel
from ncc.dataset import UnilangDataloader
from ncc.metric import BaseLoss
from ncc.utils.util_data import batch_to_cuda
from ncc.eval.evaluator import Evaluator
from tabulate import tabulate
from typing import Dict

class ACTrainer(Trainer):
    '''
    Actor-Critic Trainer
    '''

    def __init__(self, args: Dict, ) -> None:
        super(ACTrainer, self).__init__(args)

    def train(self, model: IModel, critic: IModel, dataset: UnilangDataloader, pg_criterion, lm_criterion: BaseLoss,
              critic_criterion, optimizer: Optimizer, critic_optimizer: Optimizer, reward_func='bleu', SAVE_DIR=None, start_time=None, ):
        super().train()
        start_time = time.time() if start_time is None else start_time

        bleu1_best, rougel_best, cider_best = 0.0, 0.0, 0.0
        for epoch in range(1, 1 + model.args['ac']['train_epoch_ac']):
            model.train(), critic.train()
            train_data_iter = iter(dataset['train'])
            total_loss = 0.0

            for iteration in range(1, 1 + len(dataset['train'])):
                batch = train_data_iter.__next__()
                if model.args['common']['device'] is not None:
                    batch = batch_to_cuda(batch)

                if model.args['training']['pointer']:
                    code_dict_comment, comment_extend_vocab, pointer_extra_zeros, code_oovs = batch['pointer']
                else:
                    code_oovs = None

                enc_output, dec_hidden, enc_mask = model.encoder.forward(batch)
                sample_opt = {'sample_max': 0, 'seq_length': model.args['training']['max_predict_length']}
                comment, comment_logprobs, comment_logp_gathered, comment_padding_mask, reward, comment_lprob_sum, dec_output, dec_hidden, \
                    = model.decoder.forward_pg(batch, enc_output, dec_hidden, enc_mask, dataset.token_dicts, sample_opt,
                                               reward_func,
                                               code_oovs)

                # critic
                enc_output_critic, dec_hidden_critic, enc_mask_critic = critic.encoder.forward(batch)
                sample_opt = {'sample_max': 0, 'seq_length': critic.args['training']['max_predict_length']}
                # comment, comment_logprobs, comment_logp_gathered, comment_padding_mask, comment_lprob_sum, dec_hidden,
                # _, _, _, _, _, _, dec_hidden_output, dec_hidden_critic, \
                #     = critic.decoder.forward_pg(batch, enc_output_critic, dec_hidden_critic, enc_mask_critic, dataset.token_dicts, sample_opt)
                comment_critic, comment_logprobs_critic, comment_logp_gathered_critic, comment_padding_mask_critic, comment_lprob_sum_critic, \
                dec_output_critic, dec_hidden_critic, = critic.decoder.forward(batch, enc_output_critic,
                                                                             dec_hidden_critic,
                                                                             enc_mask_critic, sample_opt)
                # print('critic: ', critic)
                value = critic.value(dec_output_critic.reshape(-1, dec_output_critic.size(-1))).view_as(reward)  # (batch_size*comment_len)
                critic_loss = critic_criterion(value, reward.detach())  # value: (batch_size*comment_len), reward: (batch_size*comment_len)
                critic_loss = critic_loss * comment_padding_mask
                critic_loss = torch.sum(critic_loss) / torch.sum(comment_padding_mask).float()  # comment.data.ne(data.Constants.PAD)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                if critic.args['ac']['max_grad_norm_critic'] != -1:
                    nn.utils.clip_grad_norm_(critic.parameters(), critic.args['ac']['max_grad_norm_critic'])
                critic_optimizer.step()

                value2 = value.detach() # .reshape(-1, 1).repeat(1, reward.size(1))  # .cuda()
                # print('reward: ', reward.size())
                # print(reward)
                # print('value2: ', value2.size())
                # print(value2)
                ac_loss = pg_criterion(comment_logprobs, comment, comment_padding_mask, reward - value2)  # -value2

                optimizer.zero_grad()
                ac_loss.backward()
                total_loss += ac_loss.item()
                if model.args['ac']['max_grad_norm'] != -1:
                    nn.utils.clip_grad_norm_(model.parameters(), model.args['ac']['max_grad_norm'])
                optimizer.step()

                if iteration % model.args['training']['log_interval'] == 0 and iteration > 0:
                    LOGGER.info('Epoch: {:0>3d}/{:0>3d}, batches: {:0>3d}/{:0>3d}, avg_loss: {:.8f}; time: {}'.format(
                        epoch, model.args['training']['train_epoch'], iteration, len(dataset['train']),
                        total_loss / iteration, str(datetime.timedelta(seconds=int(time.time() - start_time)))))

            bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider = \
                Evaluator.summarization_eval(model, dataset['valid'], dataset.token_dicts, lm_criterion)
            headers = ['B1', 'B2', 'B3', 'B4', 'Meteor', 'R1', 'R2', 'R3', 'R4', 'RL', 'Cider']
            result_table = [[round(i, 4) for i in [bleu1, bleu2, bleu3, bleu4, meteor,
                                                   rouge1, rouge2, rouge3, rouge4, rougel, cider]]]
            LOGGER.info('Evaluation results:\n{}'.format(tabulate(result_table, headers=headers,
                                                                  tablefmt=model.args['common'][
                                                                      'result_table_format'])))

            if SAVE_DIR is not None:
                model_name_prefix = '{}-bs{}-lr{}-attn{}-pointer{}-tt{}'.format(
                    '8'.join(model.args['training']['code_modalities']),
                    model.args['training']['batch_size'],
                    model.args['ac']['lr'],
                    model.args['training']['attn_type'],
                    model.args['training']['pointer'],
                    self.__class__.__name__)

                model_name = '{}-ep{}'.format(model_name_prefix, epoch)
                model_path = os.path.join(SAVE_DIR, '{}.pt'.format(model_name), )
                torch.save(model.state_dict(), model_path)
                LOGGER.info('Dumping ac model in {}'.format(model_path))

                if 'bleu' in self.args['testing']['metrics']:
                    if bleu1 > bleu1_best:
                        bleu1_best = bleu1
                        model_path = os.path.join(SAVE_DIR, '{}-best-bleu1.pt'.format(model_name_prefix), )
                        torch.save(model.state_dict(), model_path)
                        LOGGER.info('Dumping best bleu1 model in {}'.format(model_path))
                if 'cider' in self.args['testing']['metrics']:
                    if cider > cider_best:
                        cider_best = cider
                        model_path = os.path.join(SAVE_DIR, '{}-best-cider.pt'.format(model_name_prefix), )
                        torch.save(model.state_dict(), model_path)
                        LOGGER.info('Dumping best cider model in {}'.format(model_path))

                if 'rouge' in self.args['testing']['metrics']:
                    if rougel > rougel_best:
                        rougel_best = rougel
                        model_path = os.path.join(SAVE_DIR, '{}-best-rougel.pt'.format(model_name_prefix), )
                        torch.save(model.state_dict(), model_path)
                        LOGGER.info('Dumping best rouge model in {}'.format(model_path))
        LOGGER.info('{} train end'.format(self))
