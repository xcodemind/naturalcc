# -*- coding: utf-8 -*-
import os
import datetime
import time
from tabulate import tabulate
import torch
from torch.optim.optimizer import Optimizer
from ncc import LOGGER
from ncc.trainer.trainer_ import Trainer
from ncc.models.template import IModel
from ncc.dataset import XlangDataloader
from ncc.metric import BaseLoss
from ncc.utils.util_data import batch_to_cuda, merge_data
from ncc.utils.util_optimizer import create_scheduler
from ncc.metric.base import LMLoss
from ncc.dataset.base import sbase_collate_fn
from ncc.eval.evaluator import Evaluator
from typing import Dict

class DTRLTrainer(Trainer):
    '''
    Deep Transfer Reinforcement Learning Trainer
    '''

    def __init__(self, args: Dict, ) -> None:
        super(DTRLTrainer, self).__init__(args)

    def train(self, model: IModel, dataset: XlangDataloader, criterion: BaseLoss, optimizer: Optimizer,
              SAVE_DIR=None, start_time=None, model_name_prefix=None, ):
        super().train()
        start_time = time.time() if start_time is None else start_time
        if model_name_prefix is not None:
            model_name_prefix += "-tt{}".format(self.__class__.__name__)
        scheduler = create_scheduler(optimizer,
                                     self.args['sl']['warmup_epochs'],
                                     self.args['sl']['warmup_factor'],
                                     self.args['sl']['lr_milestones'],
                                     self.args['sl']['lr_gamma'])

        lm_criterion = LMLoss(device=self.args['common']['device'] is not None, )
        src_lngs = self.args['dataset']['source']['dataset_lng']
        trg_lng = self.args['dataset']['target']['dataset_lng'][0]
        valid_dataset = dataset['trg'][trg_lng]
        # last batch data size may be different
        min_loader_len = min([len(dataset['source'][lng]['train']) for lng in src_lngs])

        bleu1_best, rougel_best, cider_best = 0.0, 0.0, 0.0
        return_model = {'bleu': None, 'cider': None, 'rouge': None}
        for epoch in range(1, 1 + self.args['training']['train_epoch']):
            model.train()
            src_train_iters = {lng: iter(dataset['src'][lng]['train']) for lng in src_lngs}
            total_loss = 0.0

            for iteration in range(1, 1 + min_loader_len):
                batch = [src_train_iters[lng].__next__() for lng in src_lngs]
                batch = merge_data(batch)
                if self.args['common']['device'] is not None:
                    batch = batch_to_cuda(batch)
                dtrl_loss = model.train_dtrl(batch, criterion, dataset.token_dicts, self.args['dtrl']['reward_func'])
                LOGGER.debug('{} batch loss: {:.8f}'.format(self.__class__.__name__, dtrl_loss.item()))
                optimizer.zero_grad()
                dtrl_loss.backward()
                total_loss += dtrl_loss.item()
                optimizer.step()

                # sometimes DTRL is too small, so print train loss when first batch train
                if iteration % self.args['training']['log_interval'] == 0 or iteration == 1:
                    LOGGER.info(
                        'Epoch: {:0>3d}/{:0>3d}, batches: {:0>3d}/{:0>3d}, avg_loss: {:.6f}; lr: {:.6f}, time: {}'. \
                            format(epoch, self.args['training']['train_epoch'], iteration, min_loader_len,
                                   total_loss / iteration, scheduler.get_lr()[0],
                                   str(datetime.timedelta(seconds=int(time.time() - start_time)))))

            scheduler.step(epoch)

            # Validation on each epoch
            bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider = \
                Evaluator.summarization_eval(model, valid_dataset['valid'], dataset.token_dicts, lm_criterion,
                                             collate_func=sbase_collate_fn)
            headers = ['B1', 'B2', 'B3', 'B4', 'Meteor', 'R1', 'R2', 'R3', 'R4', 'RL', 'Cider']
            result_table = [[round(i, 4) for i in [bleu1, bleu2, bleu3, bleu4, meteor,
                                                   rouge1, rouge2, rouge3, rouge4, rougel, cider]]]
            LOGGER.info('Evaluation results:\n{}'.format(tabulate(result_table, headers=headers,
                                                                  tablefmt=model.args['common'][
                                                                      'result_table_format'])))

            # Dump the model if save_dir exists.
            if SAVE_DIR is not None:
                if model_name_prefix is None:
                    model_name_prefix = '{}-bs{}-lr{}-attn{}-pointer{}-tt{}'.format(
                        '8'.join(self.args['training']['code_modalities']),
                        self.args['training']['batch_size'],
                        self.args['sl']['lr'],
                        self.args['training']['attn_type'],
                        self.args['training']['pointer'],
                        self.__class__.__name__)

                model_name = '{}-ep{}'.format(model_name_prefix, epoch)
                model_path = os.path.join(SAVE_DIR, '{}.pt'.format(model_name), )
                torch.save(model.state_dict(), model_path)
                LOGGER.info('Dumping model in {}'.format(model_path))

                if 'bleu' in self.args['testing']['metrics']:
                    if bleu1 > bleu1_best:
                        bleu1_best = bleu1
                        model_path = os.path.join(SAVE_DIR, '{}-best-bleu1.pt'.format(model_name_prefix), )
                        return_model['bleu'] = model_path
                        torch.save(model.state_dict(), model_path)
                        LOGGER.info('Dumping best bleu1 model in {}'.format(model_path))
                if 'cider' in self.args['testing']['metrics']:
                    if cider > cider_best:
                        cider_best = cider
                        model_path = os.path.join(SAVE_DIR, '{}-best-cider.pt'.format(model_name_prefix), )
                        return_model['cider'] = model_path
                        torch.save(model.state_dict(), model_path)
                        LOGGER.info('Dumping best cider model in {}'.format(model_path))

                if 'rouge' in self.args['testing']['metrics']:
                    if rougel > rougel_best:
                        rougel_best = rougel
                        model_path = os.path.join(SAVE_DIR, '{}-best-rougel.pt'.format(model_name_prefix), )
                        return_model['rouge'] = model_path
                        torch.save(model.state_dict(), model_path)
                        LOGGER.info('Dumping best rouge model in {}'.format(model_path))

        LOGGER.info('{} train end'.format(self))
        return return_model
