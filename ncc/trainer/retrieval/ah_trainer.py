# -*- coding: utf-8 -*-
import os
import time
import datetime
from tabulate import tabulate
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from ncc import LOGGER
from ncc.trainer.trainer_ import Trainer
from ncc.models.template import IModel
from ncc.dataset import UnilangDataloader
from ncc.metric import BaseLoss
from ncc.utils.util_data import batch_to_cuda
from ncc.eval.evaluator import Evaluator
from ncc.utils.util_optimizer import create_scheduler
from typing import Dict


class AHTrainer(Trainer):
    '''
    Adversarial Hashing Learning Trainer
    '''

    def __init__(self, config: Dict, ) -> None:
        super(AHTrainer, self).__init__(config)

    def train_al(self, model: IModel, dataset: UnilangDataloader, criterion: BaseLoss, optimizer: Optimizer,
                 disc_optimizer: Optimizer, SAVE_DIR=None, start_time=None, ):
        super().train_al()
        start_time = time.time() if start_time is None else start_time

        scheduler = create_scheduler(optimizer,
                                     self.config['sl']['warmup_epochs'],
                                     self.config['sl']['warmup_factor'],
                                     self.config['sl']['lr_milestones'],
                                     self.config['sl']['lr_gamma'])
        for epoch in range(1, 1 + self.config['training']['train_epoch']):
            model.train()
            train_data_iter = iter(dataset['train'])
            total_loss = 0.0

            for iteration in range(1, 1 + len(dataset['train'])):
                batch = train_data_iter.__next__()
                if self.config['common']['device'] is not None:
                    batch = batch_to_cuda(batch)
                al_loss = model.train_al(batch, criterion, disc_optimizer)
                optimizer.zero_grad()
                al_loss.backward()
                total_loss += al_loss.item()
                if model.config['sl']['max_grad_norm'] != -1:
                    nn.utils.clip_grad_norm_(model.parameters(), model.config['sl']['max_grad_norm'])
                optimizer.step()

                if iteration % self.config['training']['log_interval'] == 0 and iteration > 0:
                    LOGGER.info(
                        'Epoch: {:0>3d}/{:0>3d}, batches: {:0>3d}/{:0>3d}, avg_loss: {:.6f}; lr: {:.6f}, time: {}'. \
                            format(epoch, self.config['training']['train_epoch'], iteration, len(dataset['train']),
                                   total_loss / iteration, scheduler.get_lr()[0],
                                   str(datetime.timedelta(seconds=int(time.time() - start_time)))))

            scheduler.step(epoch)

            # Validation on each epoch
            acc, mmr, map, ndcg, pool_size = Evaluator.retrieval_eval(model, dataset['valid'], pool_size=1000)

            headers = ['ACC@{}'.format(pool_size), 'MRR@{}'.format(pool_size), 'MAP@{}'.format(pool_size),
                       'NDCG@{}'.format(pool_size), ]
            result_table = [[round(i, 4) for i in [acc, mmr, map, ndcg]]]
            LOGGER.info('Evaluation results:\n{}'.format(tabulate(result_table, headers=headers,
                                                                  tablefmt=model.config['common'][
                                                                      'result_table_format'])))

            # Dump the model if save_dir exists.
            if SAVE_DIR is not None:
                model_name_prefix = '{}-bs{}-lr{}-attn{}'.format(
                    '8'.join(self.config['training']['code_modalities']),
                    self.config['training']['batch_size'],
                    self.config['sl']['lr'],
                    self.config['training']['attn_type'],
                )
                model_name = '{}-ep{}'.format(model_name_prefix, epoch)
                model_path = os.path.join(SAVE_DIR, '{}.pt'.format(model_name), )
                torch.save(model.state_dict(), model_path)
                LOGGER.info('Dumping model in {}'.format(model_path))

        LOGGER.info('{} train end'.format(self))

    def train_hash(self, model: IModel, dataset: UnilangDataloader, code_optimizer: Optimizer,
                   cmnt_optimizer: Optimizer, SAVE_DIR=None, start_time=None, ):
        start_time = time.time() if start_time is None else start_time

        code_scheduler = create_scheduler(code_optimizer,
                                          self.config['hash']['warmup_epochs'],
                                          self.config['hash']['warmup_factor'],
                                          self.config['hash']['lr_milestones'],
                                          self.config['hash']['lr_gamma'])
        cmnt_scheduler = create_scheduler(cmnt_optimizer,
                                          self.config['hash']['warmup_epochs'],
                                          self.config['hash']['warmup_factor'],
                                          self.config['hash']['lr_milestones'],
                                          self.config['hash']['lr_gamma'])

        for epoch in range(1, 1 + self.config['training']['train_epoch']):
            model.code_encoder.train()
            model.hash_encoder.train()

            # train code
            train_data_iter = iter(dataset['train'])
            total_loss = 0.0
            batch = train_data_iter.__next__()
            if self.config['common']['device'] is not None:
                batch = batch_to_cuda(batch)
            for iteration in range(1, len(dataset['train'])):
                # batch = train_data_iter.__next__()
                # if self.config['common']['device'] is not None:
                #     batch = batch_to_cuda(batch)
                batch_loss = model.train_code_layers(batch, code_optimizer)
                total_loss += batch_loss.item()
            code_scheduler.step(epoch)
            exit()

            model.comment_encoder.train()
            model.hash_encoder.train()
            # train comment
            train_data_iter = iter(dataset['train'])
            total_loss = 0.0
            for iteration in range(1, len(dataset['train'])):
                batch = train_data_iter.__next__()
                if self.config['common']['device'] is not None:
                    batch = batch_to_cuda(batch)
                batch_loss = model.train_comment_layers(batch, cmnt_optimizer)
                total_loss += batch_loss.item()
            cmnt_scheduler.step(epoch)

            model.update_hash_mat()
            val_loss = model.calc_total_loss()
            LOGGER.info('val_loss: {:.4f}'.format(val_loss.item()))

            train_data_iter = iter(dataset['train'])
            total_loss = 0.0
            for iteration in range(1, len(dataset['train'])):
                batch = train_data_iter.__next__()
                if self.config['common']['device'] is not None:
                    batch = batch_to_cuda(batch)
                total_loss += model.hamming_loss(batch,
                                                 print_hash=iteration == (len(dataset['train']) - 1)).item()
            total_loss /= len(dataset['train']) - 1
            LOGGER.info('H loss: {:.4f}'.format(total_loss))
