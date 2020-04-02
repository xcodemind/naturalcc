# -*- coding: utf-8 -*-

import sys

sys.path.append('./')

from ncc import *
from ncc.trainer import *
from ncc.model import *
from ncc.model.template import *
from ncc.dataset import *
from ncc.metric import *
from ncc.utils.util_data import batch_to_cuda
from ncc.utils.util_eval import *
from ncc.eval import *
from ncc.utils.util_optimizer import create_scheduler
from ncc.trainer.summarization import SLTrainer
from ncc.metric.base import *


class MAMLTrainer(Trainer):
    '''
    model-agnostic meta learning
    '''

    def __init__(self, config: Dict, ) -> None:
        super(MAMLTrainer, self).__init__(config)
        self.sl_trainer = SLTrainer(config)

    def meta_train(self, model: IModel,
                   meta_train_loader: DataLoader, meta_train_size: int,
                   meta_criterion: BaseLoss, meta_optimizer: Optimizer, ) -> None:
        # meta train is to train a meta learner for a task
        train_iter = iter(meta_train_loader)
        if meta_train_size == -1:
            meta_train_size = len(meta_train_loader)
        else:
            meta_train_size = min(len(meta_train_loader), meta_train_size)

        for _ in range(1, 1 + meta_train_size):
            # load batch data
            meta_train_batch = train_iter.__next__()
            if model.config['common']['device'] is not None:
                meta_train_batch = batch_to_cuda(meta_train_batch)

            comment_loss = model.train_sl(meta_train_batch, meta_criterion)
            meta_optimizer.zero_grad()
            comment_loss.backward()
            meta_optimizer.step()

    def meta_val(self, model: IModel,
                 meta_val_loader: DataLoader, meta_val_size: int,
                 meta_criterion: BaseLoss, ) -> Any:
        if meta_val_size == -1:
            meta_val_size = len(meta_val_loader)
        else:
            meta_val_size = min(len(meta_val_loader), meta_val_size)

        val_iter = iter(meta_val_loader)

        # first batch
        # only save first graph. such method can save lot os cuda memory
        model.train()
        meta_val_batch = val_iter.__next__()
        if model.config['common']['device'] is not None:
            meta_val_batch = batch_to_cuda(meta_val_batch)
        meta_val_loss = model.train_sl(meta_val_batch, meta_criterion)

        with torch.no_grad():
            for _ in range(1, meta_val_size):
                meta_val_batch = val_iter.__next__()
                if model.config['common']['device'] is not None:
                    meta_val_batch = batch_to_cuda(meta_val_batch)
                batch_meta_val_loss = model.train_sl(meta_val_batch, meta_criterion)
                meta_val_loss += batch_meta_val_loss.item()
                # torch.cuda.empty_cache()
        return meta_val_loss / meta_val_size

    def train(self, model: IModel, dataset: XlangDataloader, criterion: BaseLoss,
              optimizer: Optimizer, meta_optimizer: Optimizer,
              SAVE_DIR=None, start_time=None, ):
        super().train()
        start_time = time.time() if start_time is None else start_time

        trg_lng = self.config['dataset']['target']['dataset_lng'][0]

        #############################################################################################
        # because meta-train is time costly, therefore we only use min batch-size of datasets
        #############################################################################################
        # if self.config['maml']['meta_epoch'] is None:
        #     meta_train_epoch = min([len(dataset['train']) for dataset in dataset['source'].values()])
        # else:
        #     meta_train_epoch = min([len(dataset['train']) for dataset in dataset['source'].values()])
        #     meta_train_epoch = min(meta_train_epoch, model.config['maml']['meta_epoch'])
        # LOGGER.info('meta train min batch num: {}'.format(meta_train_epoch))
        meta_train_size = self.config['maml']['meta_train_size']
        meta_val_size = self.config['maml']['meta_val_size']
        #############################################################################################

        for epoch in range(1, 1 + self.config['training']['train_epoch']):
            model.train()
            total_loss = 0.0

            ori_weights = deepcopy(model.state_dict())
            for task_ind, (task_name, task_item) in enumerate(dataset['source'].items()):
                #######################################################################################
                # meta train over different datasets
                # LOGGER.info('meta train on {}-train'.format(task_item))
                self.meta_train(model, task_item['train'], meta_train_size, criterion, meta_optimizer)
                #######################################################################################

                #######################################################################################
                # gradient by gradient
                # LOGGER.info('meta valid on {}-valid'.format(task_item))
                meta_val_loss = self.meta_val(model, task_item['valid'], meta_val_size, criterion)
                #######################################################################################

                LOGGER.info('meta-train task of {} loss: {:.4f}'.format(task_name, meta_val_loss.item()))
                total_loss += meta_val_loss
                # model.load_state_dict({name: ori_weights[name] for name in ori_weights})
                model.load_state_dict(ori_weights)
            total_loss /= len(dataset['source'])
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            LOGGER.info('Epoch: {:0>3d}/{:0>3d}, avg_loss: {:.4f}; time: {}'.format(
                epoch, self.config['training']['train_epoch'], total_loss.item(),
                str(datetime.timedelta(seconds=int(time.time() - start_time)))))

            if epoch <= self.config['training']['train_epoch'] and epoch % 10 == 0:
                if SAVE_DIR is not None:
                    model_name = '{}-bs{}-{}({})-m{}({})-EPOCH{}-{}-{}'.format(
                        '8'.join(self.config['training']['code_modalities']),
                        self.config['training']['batch_size'],
                        self.config['sl']['optim'], self.config['sl']['lr'],
                        self.config['maml']['meta_optim'], self.config['maml']['meta_lr'],
                        self.config['maml']['meta_train_size'], self.config['maml']['meta_val_size'], epoch)
                    model_path = os.path.join(SAVE_DIR, '{}.pt'.format(model_name), )
                    torch.save(model.state_dict(), model_path)
                    LOGGER.info('Dumping model in {}'.format(model_path))

                ori_weights = deepcopy(model.state_dict())
                # finetune on target train dataset, but DO NOT update model
                tmp_optimizer = getattr(torch.optim, self.config['sl']['optim']) \
                    (model.parameters(), self.config['sl']['lr'])
                self.finetune(model, dataset['target'][trg_lng], criterion, tmp_optimizer)
                model.load_state_dict(ori_weights)
            else:
                pass
        LOGGER.info('{} train end'.format(self))

    def finetune(self, model: IModel,
                 dataset: UnilangDataloader,
                 criterion: BaseLoss, optimizer: Optimizer, ):
        '''
        finetune modal only on a batch data, to check out
        '''
        lm_criterion = LMLoss(device=self.config['common']['device'] is not None, )

        cider_list = []
        # zero-shot
        metrics = Evaluator.summarization_eval(model, dataset['test'], dataset.token_dicts, lm_criterion,
                                               metrics=['cider'], )
        cider_list.append(round(metrics[-1], 4))
        # few-shot
        LOGGER.info('finetune on {} '.format(dataset.lng))
        for epoch in range(1, 1 + self.config['maml']['mini_finetune_epoch']):
            model.train()
            train_data_iter = iter(dataset['train'])
            for _ in range(1, 1 + len(train_data_iter)):
                batch = train_data_iter.__next__()
                if self.config['common']['device'] is not None:
                    batch = batch_to_cuda(batch)

                sl_loss = model.train_sl(batch, criterion)
                optimizer.zero_grad()
                sl_loss.backward()
                optimizer.step()
            metrics = Evaluator.summarization_eval(model, dataset['test'], dataset.token_dicts, lm_criterion,
                                                   metrics=['cider'], )
            cider_list.append(round(metrics[-1], 4))
        LOGGER.info('Cider: {}, max: {:.4f}, mean: {:.4f}'.format(cider_list, max(cider_list),
                                                                  sum(cider_list) / len(cider_list)))
