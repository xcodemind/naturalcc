# -*- coding: utf-8 -*-
# ref: https://github.com/eric-xw/AREL
# ref: https://github.com/as10896/STC-Conditional-SeqGAN-Pytorch
import os
import datetime
import time
import torch
from torch.optim.optimizer import Optimizer
from ncc import LOGGER
from ncc.trainer.trainer_ import Trainer
from ncc.model.template import IModel
from ncc.dataset import UnilangDataloader
from ncc.metric import BaseLoss
from ncc.utils.util_data import batch_to_cuda
from ncc.utils.util_gan import AlterFlag
from typing import Dict

class GANTrainer(Trainer):
    '''
    Generative Adversarial Network Trainer
    '''

    def __init__(self, args: Dict, ) -> None:
        super(GANTrainer, self).__init__(args)

    def train(self, model: IModel, disc: IModel, dataset: UnilangDataloader, pg_criterion, lm_criterion: BaseLoss,
              disc_criterion, optimizer: Optimizer, disc_optimizer: Optimizer, SAVE_DIR=None, start_time=None, ):
        super().train()
        start_time = time.time() if start_time is None else start_time
        alter_flag = AlterFlag(D_iters=args['D_iter'], G_iters=args['G_iter'], always=args['always'])

        for epoch in range(1, 1 + model.args['rl']['train_epoch_gan']):
            model.train(), disc.train()
            train_data_iter = iter(dataset['train'])
            total_loss = 0.0

            for iteration in range(1, 1 + len(dataset['train'])):
                batch = train_data_iter.__next__()
                if model.args['common']['device'] is not None:
                    batch = batch_to_cuda(batch)

                if model.args['pointer']:
                    code_dict_comment, comment_extend_vocab, pointer_extra_zeros, code_oovs = batch['pointer']
                else:
                    code_oovs = None

                if alter_flag.flag == 'disc':
                    enc_output_critic, dec_hidden_critic, enc_mask_critic = disc.code_encoder.forward(batch)
                    sample_opt = {'sample_max': 0, 'seq_length': disc.args['max_predict_length']}
                    # comment, comment_logprobs, comment_logp_gathered, comment_padding_mask, comment_lprob_sum, dec_hidden,
                    # comment_critic, comment_logprobs_critic, comment_logp_gathered_critic, comment_padding_mask_critic, comment_lprob_sum, \
                    # dec_output_critic, dec_hidden_critic, = self.decoder.forward(batch, enc_output_critic, dec_hidden_critic,
                    #                                                              enc_mask_critic, sample_opt)
                    xxx = disc.comment_encoder.forward()

                    # value = self.value(dec_output_critic.reshape(-1, dec_output_critic.size(-1))).view_as(reward) # (batch_size*comment_len)
                    # self.proj
                    # self.fc
                    # self.activation
                    disc_loss = criterion(value, label)
                elif alter_flag.flag == 'gan':
                    comment, comment_input, comment_target, comment_len, raw_comment = batch['comment']
                    enc_output, dec_hidden, enc_mask = model.encoder.forward(batch)
                    sample_opt = {'sample_max': 0, 'seq_length': critic.args['max_predict_length']}
                    # comment, comment_logprobs, comment_logp_gathered, comment_padding_mask, comment_lprob_sum, dec_hidden,
                    # _, _, _, _, _, _, dec_hidden_output, dec_hidden_critic, \
                    #     = critic.decoder.forward_pg(batch, enc_output_critic, dec_hidden_critic, enc_mask_critic, dataset.token_dicts, sample_opt)
                    comment_gen, comment_logprobs, comment_logp_gathered, comment_padding_mask, comment_lprob_sum, \
                    dec_output, dec_hidden, = model.decoder.forward(batch, enc_output, dec_hidden, enc_mask, sample_opt)
                    # print('critic: ', critic)

                    reward = disc()
                    gen_loss = pg_criterion(comment_logprobs, comment_gen, comment_padding_mask, reward)  # -value2

                    optimizer.zero_grad()
                    gen_loss.backward()
                    total_loss += gen_loss.item()
                    optimizer.step()

                if iteration % model.args['training']['log_interval'] == 0 and iteration > 0:
                    LOGGER.info('Epoch: {:0>3d}/{:0>3d}, batches: {:0>3d}/{:0>3d}, avg_loss: {:.8f}; time: {}'.format(
                        epoch, model.args['all_epoch'], iteration, len(dataset['train']), total_loss / iteration,
                        str(datetime.timedelta(seconds=int(time.time() - start_time)))))

            if epoch <= model.args['gan']['train_epoch_gan']:
                if SAVE_DIR is not None:
                    model_name = '{}-bs{}-lr{}-attn{}-pointer{}-ep{}'.format('8'.join(model.args['code_modalities']),
                                                                                    model.args['batch_size'],
                                                                                    model.args['rl']['lr_critic'],
                                                                                    model.args['attn_type'],
                                                                                    model.args['pointer'], epoch)
                    model_path = os.path.join(SAVE_DIR, '{}.pt'.format(model_name), )
                    torch.save(model.state_dict(), model_path)
                    LOGGER.info('Dumping gan model in {}'.format(model_path))
                # Evaluator.summarization_eval(critic, dataset['valid'], dataset.token_dicts, )
            else:
                pass
        LOGGER.info('{} train end'.format(self))
