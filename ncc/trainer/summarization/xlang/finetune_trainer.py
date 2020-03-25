# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from ncc import *
from ncc.trainer import *
from ncc.trainer import *
from ncc.model import *
from ncc.model.template import *
from ncc.dataset import *
from ncc.metric import *
from ncc.utils.util_data import batch_to_cuda
from ncc.utils.util_eval import *
from ncc.eval import *
from ncc.utils.util_eval import *
from ncc.trainer.summarization.unilang.sl_trainer import SLTrainer
from ncc.utils.util_optimizer import create_scheduler
from tabulate import tabulate


class FTTrainer(Trainer):
    '''
    fine-tune transfer learning
    fine-tune needs a trainer
    '''

    def __init__(self, config: Dict, ) -> None:
        super(FTTrainer, self).__init__(config)
        self.sl_trainer = SLTrainer(config)

    def pretrain(self, model: IModel, dataset: UnilangDataloader, criterion: BaseLoss, optimizer: Optimizer,
                 SAVE_DIR=None, start_time=None, model_name_prefix=None, scheduler=None) -> Any:
        super().train()
        start_time = time.time() if start_time is None else start_time
        if model_name_prefix is not None:
            model_name_prefix += "-tt{}".format(self.__class__.__name__)
        if scheduler is None:
            scheduler = create_scheduler(optimizer,
                                         self.config['sl']['warmup_epochs'],
                                         self.config['sl']['warmup_factor'],
                                         self.config['sl']['lr_milestones'],
                                         self.config['sl']['lr_gamma'])

        bleu1_best, rougel_best, cider_best = 0.0, 0.0, 0.0
        return_model = {'bleu': None, 'cider': None, 'rouge': None}
        for epoch in range(1, 1 + self.config['finetune']['pretrain_epoch']):
            model.train()
            train_data_iter = iter(dataset['train'])
            total_loss = 0.0

            for iteration in range(1, 1 + len(dataset['train'])):
                batch = train_data_iter.__next__()
                if self.config['common']['device'] is not None:
                    batch = batch_to_cuda(batch)
                sl_loss = model.train_sl(batch, criterion)
                LOGGER.debug('{} batch loss: {:.8f}'.format(self.__class__.__name__, sl_loss.item()))
                optimizer.zero_grad()
                sl_loss.backward()
                total_loss += sl_loss.item()
                if model.config['sl']['max_grad_norm'] != -1:
                    nn.utils.clip_grad_norm_(model.parameters(), model.config['sl']['max_grad_norm'])
                optimizer.step()

                if iteration % self.config['training']['log_interval'] == 0 and iteration > 0:
                    LOGGER.info(
                        'Epoch: {:0>3d}/{:0>3d}, batches: {:0>3d}/{:0>3d}, avg_loss: {:.6f}; lr: {:.6f}, time: {}'. \
                            format(epoch, self.config['finetune']['pretrain_epoch'], iteration, len(dataset['train']),
                                   total_loss / iteration, scheduler.get_lr()[0],
                                   str(datetime.timedelta(seconds=int(time.time() - start_time)))))

            scheduler.step(epoch)

            # Validation on each epoch
            bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider = \
                Evaluator.summarization_eval(model, dataset['valid'], dataset.token_dicts, criterion)
            headers = ['B1', 'B2', 'B3', 'B4', 'Meteor', 'R1', 'R2', 'R3', 'R4', 'RL', 'Cider']
            result_table = [[round(i, 4) for i in [bleu1, bleu2, bleu3, bleu4, meteor,
                                                   rouge1, rouge2, rouge3, rouge4, rougel, cider]]]
            LOGGER.info('Evaluation results:\n{}'.format(tabulate(result_table, headers=headers,
                                                                  tablefmt=model.config['common'][
                                                                      'result_table_format'])))

            # Dump the model if save_dir exists.
            if SAVE_DIR is not None:
                if model_name_prefix is None:
                    model_name_prefix = '{}-bs{}-lr{}-attn{}-pointer{}-tt{}'.format(
                        '8'.join(self.config['training']['code_modalities']),
                        self.config['training']['batch_size'],
                        self.config['sl']['lr'],
                        self.config['training']['attn_type'],
                        self.config['training']['pointer'],
                        self.__class__.__name__)

                model_name = '{}-ep{}'.format(model_name_prefix, epoch)
                model_path = os.path.join(SAVE_DIR, '{}.pt'.format(model_name), )
                torch.save(model.state_dict(), model_path)
                LOGGER.info('Dumping model in {}'.format(model_path))

                if 'bleu' in self.config['testing']['metrics']:
                    if bleu1 > bleu1_best:
                        bleu1_best = bleu1
                        model_path = os.path.join(SAVE_DIR, '{}-best-bleu1.pt'.format(model_name_prefix), )
                        return_model['bleu'] = model_path
                        torch.save(model.state_dict(), model_path)
                        LOGGER.info('Dumping best bleu1 model in {}'.format(model_path))
                if 'cider' in self.config['testing']['metrics']:
                    if cider > cider_best:
                        cider_best = cider
                        model_path = os.path.join(SAVE_DIR, '{}-best-cider.pt'.format(model_name_prefix), )
                        return_model['cider'] = model_path
                        torch.save(model.state_dict(), model_path)
                        LOGGER.info('Dumping best cider model in {}'.format(model_path))

                if 'rouge' in self.config['testing']['metrics']:
                    if rougel > rougel_best:
                        rougel_best = rougel
                        model_path = os.path.join(SAVE_DIR, '{}-best-rougel.pt'.format(model_name_prefix), )
                        return_model['rouge'] = model_path
                        torch.save(model.state_dict(), model_path)
                        LOGGER.info('Dumping best rouge model in {}'.format(model_path))

        LOGGER.info('{} train end'.format(self))

    def finetune(self, model: IModel, dataset: UnilangDataloader, criterion: BaseLoss, optimizer: Optimizer,
                 SAVE_DIR=None, start_time=None, model_name_prefix=None, scheduler=None) -> Any:
        super().train()
        start_time = time.time() if start_time is None else start_time
        if model_name_prefix is not None:
            model_name_prefix += "-tt{}".format(self.__class__.__name__)
        if scheduler is None:
            scheduler = create_scheduler(optimizer,
                                         self.config['sl']['warmup_epochs'],
                                         self.config['sl']['warmup_factor'],
                                         self.config['sl']['lr_milestones'],
                                         self.config['sl']['lr_gamma'])

        bleu1_best, rougel_best, cider_best = 0.0, 0.0, 0.0
        return_model = {'bleu': None, 'cider': None, 'rouge': None}
        for epoch in range(1, 1 + self.config['training']['train_epoch']):
            model.train()
            train_data_iter = iter(dataset['train'])
            total_loss = 0.0

            for iteration in range(1, 1 + len(dataset['train'])):
            # for iteration in range(1, 1 + 10):
                batch = train_data_iter.__next__()
                if self.config['common']['device'] is not None:
                    batch = batch_to_cuda(batch)
                sl_loss = model.train_sl(batch, criterion)
                LOGGER.debug('{} batch loss: {:.8f}'.format(self.__class__.__name__, sl_loss.item()))
                optimizer.zero_grad()
                sl_loss.backward()
                total_loss += sl_loss.item()
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
            bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider = \
                Evaluator.summarization_eval(model, dataset['test'], dataset.token_dicts, criterion)
            headers = ['B1', 'B2', 'B3', 'B4', 'Meteor', 'R1', 'R2', 'R3', 'R4', 'RL', 'Cider']
            result_table = [[round(i, 4) for i in [bleu1, bleu2, bleu3, bleu4, meteor,
                                                   rouge1, rouge2, rouge3, rouge4, rougel, cider]]]
            LOGGER.info('Evaluation results:\n{}'.format(tabulate(result_table, headers=headers,
                                                                  tablefmt=model.config['common'][
                                                                      'result_table_format'])))

            # Dump the model if save_dir exists.
            if SAVE_DIR is not None:
                if model_name_prefix is None:
                    model_name_prefix = '{}-bs{}-lr{}-attn{}-pointer{}-tt{}'.format(
                        '8'.join(self.config['training']['code_modalities']),
                        self.config['training']['batch_size'],
                        self.config['sl']['lr'],
                        self.config['training']['attn_type'],
                        self.config['training']['pointer'],
                        self.__class__.__name__)

                model_name = '{}-ep{}'.format(model_name_prefix, epoch)
                model_path = os.path.join(SAVE_DIR, '{}.pt'.format(model_name), )
                torch.save(model.state_dict(), model_path)
                LOGGER.info('Dumping model in {}'.format(model_path))

                if 'bleu' in self.config['testing']['metrics']:
                    if bleu1 > bleu1_best:
                        bleu1_best = bleu1
                        model_path = os.path.join(SAVE_DIR, '{}-best-bleu1.pt'.format(model_name_prefix), )
                        return_model['bleu'] = model_path
                        torch.save(model.state_dict(), model_path)
                        LOGGER.info('Dumping best bleu1 model in {}'.format(model_path))
                if 'cider' in self.config['testing']['metrics']:
                    if cider > cider_best:
                        cider_best = cider
                        model_path = os.path.join(SAVE_DIR, '{}-best-cider.pt'.format(model_name_prefix), )
                        return_model['cider'] = model_path
                        torch.save(model.state_dict(), model_path)
                        LOGGER.info('Dumping best cider model in {}'.format(model_path))

                if 'rouge' in self.config['testing']['metrics']:
                    if rougel > rougel_best:
                        rougel_best = rougel
                        model_path = os.path.join(SAVE_DIR, '{}-best-rougel.pt'.format(model_name_prefix), )
                        return_model['rouge'] = model_path
                        torch.save(model.state_dict(), model_path)
                        LOGGER.info('Dumping best rouge model in {}'.format(model_path))

        LOGGER.info('{} train end'.format(self))
        return return_model
