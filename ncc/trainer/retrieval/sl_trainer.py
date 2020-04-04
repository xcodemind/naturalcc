# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from ncc import *
from ncc.trainer import *
from ncc.models import *
from ncc.models.template import *
from ncc.dataset import *
from ncc.metric import *
from ncc.utils.util_data import batch_to_cuda
from ncc.utils.util_eval import *
from ncc.eval import *
from ncc.utils.util_optimizer import create_scheduler
from torch.optim import lr_scheduler
from tabulate import tabulate


class SLTrainer(Trainer):
    '''
    Supervise Learning Trainer
    '''

    def __init__(self, config: Dict, ) -> None:
        super(SLTrainer, self).__init__(config)

    def train(self, model: IModel, dataset: UnilangDataloader, criterion: BaseLoss, optimizer: Optimizer,
              SAVE_DIR=None, start_time=None, ):
        super().train()
        start_time = time.time() if start_time is None else start_time

        scheduler = create_scheduler(optimizer,
                                     self.config['sl']['warmup_epochs'],
                                     self.config['sl']['warmup_factor'],
                                     self.config['sl']['lr_milestones'],
                                     self.config['sl']['lr_gamma'])

        mmr_best, ndcg_best = 0.0, 0.0
        return_model = {'mmr': None, 'ndcg': None, }
        for epoch in range(1, 1 + self.config['training']['train_epoch']):
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
                            format(epoch, self.config['training']['train_epoch'], iteration, len(dataset['train']),
                                   total_loss / iteration, scheduler.get_lr()[0],
                                   str(datetime.timedelta(seconds=int(time.time() - start_time)))))

            scheduler.step(epoch)

            # Validation on each epoch
            _, mmr, _, ndcg, pool_size = Evaluator.retrieval_eval(model, dataset['valid'], pool_size=1000)
            # LOGGER.info('ACC@{}: {:.4f}, MRR@{}: {:.4f}, MAP@{}: {:.4f}, NDCG@{}: {:.4f}'.format(
            #     pool_size, acc, pool_size, mmr, pool_size, map, pool_size, ndcg))

            # headers = ['ACC@{}'.format(pool_size), 'MRR@{}'.format(pool_size), 'MAP@{}'.format(pool_size),
            #            'NDCG@{}'.format(pool_size), ]
            # result_table = [[round(i, 4) for i in [acc, mmr, map, ndcg]]]
            # LOGGER.info('Evaluation results:\n{}'.format(tabulate(result_table, headers=headers,
            #                                                       tablefmt=model.config['common'][
            #                                                           'result_table_format'])))

            headers = ['MRR@{}'.format(pool_size), 'NDCG@{}'.format(pool_size), ]
            result_table = [[round(i, 4) for i in [mmr, ndcg]]]
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

                if mmr > mmr_best:
                    mmr_best = mmr
                    return_model['mmr'] = model_path
                    LOGGER.info('Dumping best mmr model in {}'.format(model_path))

                if ndcg > ndcg_best:
                    ndcg_best = ndcg
                    return_model['ndcg'] = model_path
                    LOGGER.info('Dumping best ndcg model in {}'.format(model_path))

        LOGGER.info('{} train end'.format(self))
        return return_model
