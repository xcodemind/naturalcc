import os
import torch
import datetime
import time
from ncc.utils.util_data import batch_to_cuda
from ncc.eval.evaluator import Evaluator
from ncc import LOGGER
from torch.optim.lr_scheduler import LambdaLR


class CodeNNSLTrainer(object):
    def __init__(self, args, model, dataset ): #eval_data
        self.args = args
        self.model = model
        self.dataset = dataset
        self.train_dataloader = dataset['train']
        self.val_dataloader = dataset['valid']

        self.last_bleu1=0
        self.best_bleu1 = 0
        self.best_bleu1_epoch = 0
        self.best_cider = 0
        self.best_cider_epoch = 0
        self.lr = 0.5
        self.optimizer = torch.optim.SGD(model.parameters(),  self.lr)
        self.decay_coeff = 0.8
        self.max_grad = 5
        lambda1 = lambda epoch: self.decay_coeff ** epoch
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda1)

        self.lng = self.args['dataset']['source']['dataset_lng'][0]

    def train(self, criterion, start_time=None, SAVE_DIR=None ):
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time

        epoch = 1
        while True :
            self.model.train()
            total_loss, report_loss, total_words, report_words =  0, 0, 0, 0
            train_data_iter = iter(self.train_dataloader)
            # while iteration < len(self.train_dataloader)-1:
            for iteration in range(1,1+ len(self.train_dataloader)):

                batch = train_data_iter.__next__()
                if self.args['common']['device'] is not None:
                    batch = batch_to_cuda(batch)
                sl_loss = self.model.train_sl(batch, criterion)
                LOGGER.debug('{} batch loss: {:.8f}'.format(self.__class__.__name__, sl_loss.item()))

                self.optimizer.zero_grad()
                sl_loss.backward()
                total_loss += sl_loss.item()

                # if self.opt.flag_clip_grad:
                #     total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.max_grad_norm)

                torch.nn.utils.clip_grad_value_(self.model.parameters(),self.max_grad) # 看论文感觉应该是直接裁剪梯度而非梯度范数

                self.optimizer.step()



                if iteration % self.args['training']['log_interval'] == 0 and iteration > 0:
                    LOGGER.info('Epoch: {} , batches: {:0>3d}/{:0>3d}, avg_loss: {:.8f}; lr: {}, time: {}'.format(
                        epoch,  iteration, len(self.train_dataloader),
                        total_loss / iteration, self.scheduler.get_lr()[0] ,
                        str(datetime.timedelta(seconds=int(time.time() - self.start_time)))))



            bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider = \
            Evaluator.summarization_eval(self.model, self.val_dataloader, self.dataset.token_dicts,
                                         criterion,model_filename=None ,metrics=['bleu','cider']  )
            LOGGER.info("Epoch: {} val_metric bleu1:{} cider:{} ".format(epoch , bleu1,cider))

            if epoch > 60 :
                if bleu1< self.last_bleu1:
                    self.scheduler.step() # 学习率衰减
                if self.scheduler.get_lr()[0] < 0.001:
                    assert False,LOGGER.info("lr below 0.001 , so stop training ")

            self.last_bleu1 = bleu1
            if bleu1 > self.best_bleu1:
                self.best_bleu1 = bleu1
                self.best_bleu1_epoch = epoch
            if cider > self.best_cider:
                self.best_cider = cider 
                self.best_cider_epoch = epoch 
            LOGGER.info("Epoch: {} best_bleu1: {} best_bleu1_epoch:{} best_cider:{} best_cider_epoch:{} ".format(
                epoch , self.best_bleu1,self.best_bleu1_epoch,self.best_cider,self.best_cider_epoch))
            model_name = os.path.join(SAVE_DIR ,"model_sl_codenn_e{}_lng_{}.pt".format(epoch,self.lng ))

            torch.save(self.model.state_dict(), model_name)
            LOGGER.info("Save model as %s" % model_name)

            epoch += 1