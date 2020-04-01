# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from ncc import *
from ncc.trainer import *
from ncc.model import *
from ncc.model.template import *
from ncc.dataset import *
from ncc.metric import *
from ncc.utils.util_data import batch_to_cuda
from ncc.utils.util_eval import *
from ncc.eval import *
from tabulate import tabulate

class PGTrainer(Trainer):
    '''
    Reinforcement Learning with Self-Critical Trainer
    Ref: Self-critical Sequence Training for Image Captioning (https://arxiv.org/abs/1612.00563)
    '''

    def __init__(self, config: Dict, ) -> None:
        super(PGTrainer, self).__init__(config)

    def train(self, model: IModel, dataset: UnilangDataloader, lm_criterion: BaseLoss, criterion: BaseLoss, optimizer: Optimizer, reward_func='bleu',
              SAVE_DIR=None, start_time=None, ):
        super().train()
        start_time = time.time() if start_time is None else start_time

        bleu1_best, rougel_best, cider_best = 0.0, 0.0, 0.0
        for epoch in range(1, 1 + model.config['training']['train_epoch']):
            model.train()
            train_data_iter = iter(dataset['train'])
            total_loss = 0.0

            for iteration in range(1, 1 + len(dataset['train'])):
                batch = train_data_iter.__next__()
                if model.config['common']['device'] is not None:
                    batch = batch_to_cuda(batch)
                pg_loss = model.train_pg(batch, criterion, dataset.token_dicts, reward_func)
                if model.config['pg']['rl_weight'] == 1.0:
                    LOGGER.debug('No teacher forcing.')
                    loss = pg_loss
                else:
                    sl_loss = model.train_sl(batch, lm_criterion)
                    loss = model.config['pg']['rl_weight'] * pg_loss + (1 - model.config['pg']['rl_weight']) * sl_loss
                LOGGER.debug('{} batch loss: {:.8f}'.format(self.__class__.__name__, loss.item()))
                optimizer.zero_grad()
                loss.backward()
                total_loss += loss.item()
                if model.config['pg']['max_grad_norm'] != -1:
                    nn.utils.clip_grad_norm_(model.parameters(), model.config['pg']['max_grad_norm'])
                optimizer.step()

                if iteration % model.config['training']['log_interval'] == 0 and iteration > 0:
                    LOGGER.info('Epoch: {:0>3d}/{:0>3d}, batches: {:0>3d}/{:0>3d}, avg_loss: {:.8f}; time: {}'.format(
                        epoch, model.config['training']['train_epoch'], iteration, len(dataset['train']), total_loss / iteration,
                        str(datetime.timedelta(seconds=int(time.time() - start_time)))))

            # Validation on each epoch
            bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider = \
                Evaluator.summarization_eval(model, dataset['valid'], dataset.token_dicts, lm_criterion)
            headers = ['B1', 'B2', 'B3', 'B4', 'Meteor', 'R1', 'R2', 'R3', 'R4', 'RL', 'Cider']
            result_table = [[round(i, 4) for i in [bleu1, bleu2, bleu3, bleu4, meteor,
                                                   rouge1, rouge2, rouge3, rouge4, rougel, cider]]]
            LOGGER.info('Evaluation results:\n{}'.format(tabulate(result_table, headers=headers,
                                                                  tablefmt=model.config['common']['result_table_format'])))

            if SAVE_DIR is not None:
                model_name_prefix = '{}-bs{}-lr{}-attn{}-pointer{}'.format(
                    '8'.join(model.config['training']['code_modalities']),
                    model.config['training']['batch_size'],
                    model.config['pg']['lr'],
                    model.config['training']['attn_type'],
                    model.config['training']['pointer'])
                model_name = '{}-ep{}'.format(model_name_prefix, epoch)

                model_path = os.path.join(SAVE_DIR, '{}.pt'.format(model_name), )
                torch.save(model.state_dict(), model_path)
                LOGGER.info('Dumping pg model in {}'.format(model_path))

                if 'bleu' in self.config['testing']['metrics']:
                    if bleu1 > bleu1_best:
                        bleu1_best = bleu1
                        model_path = os.path.join(SAVE_DIR, '{}-best-bleu1.pt'.format(model_name_prefix), )
                        torch.save(model.state_dict(), model_path)
                        LOGGER.info('Dumping best bleu1 model in {}'.format(model_path))
                if 'cider' in self.config['testing']['metrics']:
                    if cider > cider_best:
                        cider_best = cider
                        model_path = os.path.join(SAVE_DIR, '{}-best-cider.pt'.format(model_name_prefix), )
                        torch.save(model.state_dict(), model_path)
                        LOGGER.info('Dumping best cider model in {}'.format(model_path))

                if 'rouge' in self.config['testing']['metrics']:
                    if rougel > rougel_best:
                        rougel_best = rougel
                        model_path = os.path.join(SAVE_DIR, '{}-best-rougel.pt'.format(model_name_prefix), )
                        torch.save(model.state_dict(), model_path)
                        LOGGER.info('Dumping best rouge model in {}'.format(model_path))

        LOGGER.info('{} train end'.format(self))
