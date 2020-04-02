import os
import datetime
import time
import torch
import torch.nn as nn
from ncc.utils.util_data import batch_to_cuda
from ncc.eval.evaluator import Evaluator
from ncc import LOGGER
from ncc.utils.util_optimizer import create_scheduler

class AstAttendGruSLTrainer(object):
    def __init__(self,config, model, dataset ): #eval_data
        self.config = config
        self.model = model
        self.dataset = dataset
        self.train_dataloader = dataset['train']
        self.val_dataloader = dataset['valid']

        self.last_bleu1=0
        self.best_bleu1 = 0
        self.best_bleu1_epoch = 0
        self.best_cider = 0
        self.best_cider_epoch = 0


        self.rnn_hidden_size_display = config['training']['rnn_hidden_size']
        self.tok_embed_size_display = config['training']['tok_embed_size']
        self.sbtao_embed_size_display = config['training']['sbtao_embed_size']

        # self.lr = 0.5
        # self.optimizer = torch.optim.Adam(model.parameters() )
        if self.config.__contains__('sl'):
            self.optimizer = getattr(torch.optim, config['sl']['optim']) \
                (self.model.parameters(), config['sl']['lr'])
            self.scheduler = create_scheduler(self.optimizer,
                                     self.config['sl']['warmup_epochs'],
                                     self.config['sl']['warmup_factor'],
                                     self.config['sl']['lr_milestones'],
                                     self.config['sl']['lr_gamma'])
            self.lr_display = config['sl']['lr']
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999),
                                          eps=1e-08, weight_decay=0 )
            self.lr_display = None
        # self.decay_coeff = 0.8
        # self.max_grad = 5
        # lambda1 = lambda epoch: self.decay_coeff ** epoch
        # self.scheduler  = LambdaLR(self.optimizer, lr_lambda=lambda1)

        self.lng = self.config['dataset']['source']['dataset_lng'][0]

        # self.evaluator = Evaluator(self.model, self.val_dataset, self.val_dataloader,  self.dict_comment, self.opt)

    def train(self, criterion,   start_time=None,SAVE_DIR=None ):
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time

        return_model = {'bleu': None, 'cider': None }
        # for epoch in range(1,1+self.all_epoch):
        for epoch in range(1, 1 + self.config['training']['train_epoch']):
            self.model.train()
            total_loss, report_loss, total_words, report_words =  0, 0, 0, 0
            train_data_iter = iter(self.train_dataloader)
            # while iteration < len(self.train_dataloader)-1:
            for iteration in range(1,1+ len(self.train_dataloader)):

                batch = train_data_iter.__next__()
                if self.config['common']['device'] is not None:
                    batch = batch_to_cuda(batch)
                sl_loss = self.model.train_sl(batch, criterion)
                LOGGER.debug('{} batch loss: {:.8f}'.format(self.__class__.__name__, sl_loss.item()))

                self.optimizer.zero_grad()
                sl_loss.backward()
                total_loss += sl_loss.item()
                self.optimizer.step()

                if iteration % self.config['training']['log_interval'] == 0 and iteration > 0:
                    LOGGER.info('Epoch: {} , batches: {:0>3d}/{:0>3d}, avg_loss: {:.8f}; lr:not display, time: {}'.format(
                        epoch,  iteration, len(self.train_dataloader),
                        total_loss / iteration,
                        str(datetime.timedelta(seconds=int(time.time() - self.start_time)))))

            if self.config.__contains__('sl'):
                self.scheduler.step(epoch)

            bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider = \
            Evaluator.summarization_eval(self.model, self.val_dataloader, self.dataset.token_dicts,
                                         criterion,model_filename=None ,metrics=['bleu','cider']  )
            LOGGER.info("Epoch: {} val_metric bleu1:{} cider:{} ".format(epoch , bleu1,cider))

            model_name_prefix = "model_sl_{}_lng_{}_t{}_s{}_c{}_lr_display{}_rh{}_te{}_se{}".format (
                self.model.__class__.__name__.lower(),self.lng ,self.config['dataset']['max_tok_len'],
                self.config['dataset']['max_sbtao_len'],self.config['dataset']['max_comment_len'],self.lr_display ,
            self.rnn_hidden_size_display  , self.tok_embed_size_display , self.sbtao_embed_size_display  )
            model_name = '{}-ep{}.pt'.format(model_name_prefix, epoch)
            model_name = os.path.join(SAVE_DIR,model_name )

            self.last_bleu1 = bleu1
            if bleu1 > self.best_bleu1:
                self.best_bleu1 = bleu1
                self.best_bleu1_epoch = epoch
                model_path = os.path.join(SAVE_DIR, '{}-best-cider.pt'.format(model_name_prefix), )
                return_model['bleu'] = model_path
                torch.save(self.model.state_dict(), model_path)
                LOGGER.info('Dumping best cider model in {}'.format(model_path))
            if cider > self.best_cider:
                self.best_cider = cider
                self.best_cider_epoch = epoch
                model_path = os.path.join(SAVE_DIR, '{}-best-cider.pt'.format(model_name_prefix), )
                return_model['cider'] = model_path
                torch.save(self.model.state_dict(), model_path)
                LOGGER.info('Dumping best cider model in {}'.format(model_path))

            LOGGER.info("Epoch: {} best_bleu1: {} best_bleu1_epoch:{} best_cider:{} best_cider_epoch:{} ".format(
                epoch , self.best_bleu1,self.best_bleu1_epoch,self.best_cider,self.best_cider_epoch))

            torch.save(self.model.state_dict(), model_name)
            LOGGER.info("Save model as %s" % model_name)

        return  return_model
