# -*- coding: utf-8 -*-
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
from typing import Dict


class CriticTrainer(Trainer):
    '''
    Critic Trainer
    '''

    def __init__(self, args: Dict, ) -> None:
        super(CriticTrainer, self).__init__(args)

    def train(self, model: IModel, critic: IModel, dataset: UnilangDataloader, criterion: BaseLoss,
              critic_optimizer: Optimizer, reward_func='bleu', SAVE_DIR=None, start_time=None, ):
        super().train()
        start_time = time.time() if start_time is None else start_time

        for epoch in range(1, 1 + model.args['ac']['train_epoch_critic']):
            model.eval()
            critic.train()
            train_data_iter = iter(dataset['train'])
            total_loss = 0.0

            for iteration in range(1, 1 + len(dataset['train'])):
                batch = train_data_iter.__next__()
                if critic.args['common']['device'] is not None:
                    batch = batch_to_cuda(batch)

                critic_loss = critic.train_sl(model, batch, criterion, dataset.token_dicts, reward_func)
                LOGGER.debug('{} batch loss: {:.8f}'.format(self.__class__.__name__, critic_loss.item()))
                critic_optimizer.zero_grad()
                critic_loss.backward()
                total_loss += critic_loss.item()
                critic_optimizer.step()

                if iteration % critic.args['training']['log_interval'] == 0 and iteration > 0:
                    LOGGER.info('Epoch: {:0>3d}/{:0>3d}, batches: {:0>3d}/{:0>3d}, avg_loss: {:.8f}; time: {}'.format(
                        epoch, critic.args['ac']['train_epoch_critic'], iteration, len(dataset['train']), total_loss / iteration,
                        str(datetime.timedelta(seconds=int(time.time() - start_time)))))

            if SAVE_DIR is not None:
                model_name = 'critic-{}-bs{}-lr{}-attn{}-pointer{}-ep{}'.format(
                    '8'.join(critic.args['training']['code_modalities']),
                    critic.args['training']['batch_size'],
                    critic.args['ac']['lr_critic'],
                    critic.args['training']['attn_type'],
                    critic.args['training']['pointer'], epoch)
                model_path = os.path.join(SAVE_DIR, '{}.pt'.format(model_name), )
                torch.save(critic.state_dict(), model_path)
                LOGGER.info('Dumping critic in {}'.format(model_path))

        LOGGER.info('{} train end'.format(self))
