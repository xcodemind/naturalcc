# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from src import *
from src.dataset.base import *
from src.dataset import *
from src.utils.util_data import batch_to_cuda
from src.utils.util_eval import *
from src.utils.util_optimizer import create_scheduler
from torch.optim import lr_scheduler
from src.eval.evaluator import Evaluator
from src.utils.util import save_json
import datetime
import os
import time
import torch
import numpy as np

from  src.utils.constants import * 
from  src.utils.util_data  import  output2topk

class KDSLTrainer(object):
    def __init__(self,args,config,model ,  dataset:KDDataloader ):
        self.args = args
        self.config = config
        self.model = model
        self.train_dataset = dataset['train_dataset']
        self.train_dataloader = dataset['train']
        self.val_dataset = dataset['valid_dataset']
        self.val_dataloader = dataset['valid']
        self.token_dicts = dataset.token_dicts

        self.kd_path = self.config['kd']['kd_path']
        self.distill = self.config['kd']['distill']
        self.distill_topk = config['kd']['distill_topk']

        self.oriname2finetune = model.config['sl']['oriname2finetune']
        self.code_modalities_str = '8'.join(  config['training']['code_modalities'])
        self.train_epoch = config['training']['train_epoch']

        self.source = self.config['kd']['source']
        self.target = self.config['kd']['target']

        LOGGER.info("self.source :{}  ".format(self.source))
        LOGGER.info("self.target :{}  ".format(self.target))

        if not self.distill and self.oriname2finetune == 'none':
            self.source = dataset.sources[0]
            self.target = dataset.targets[0]
        # only used in individual model train(before distill),so dataset.sources and dataset.targets has only one element
        self.trainer_type = args.train_mode
        self.best_bleu1= 0
        self.best_bleu1_epoch=0
        self.best_cider=0
        self.best_cider_epoch = 0
        self.last_best_bleu1_dict = {}

        # self.source_epoch_str = "_".join([str(i) for i in self.config['kd']['sources_epoch']]) \
        #     if self.config['kd']['sources_epoch'] is not None else None

        if self.config['kd']['sources_epoch'] is not None:
            self.source_epoch_str = ''
            for k,v in self.config['kd']['sources_epoch'].items():
                self.source_epoch_str += '_'+str(k)+str(v)
        else:
            self.source_epoch_str = None

        LOGGER.info("source_epoch_str: {} ".format(self.source_epoch_str))
        # self.all_epoch = config['training']['all_epoch']

        # if self.distill:
        #     self.evaluator = Evaluator(self.model, self.val_dataset, self.val_dataloader, (self.dict_code, self.dict_comment),
        #                           train_dataset_names=self.train_dataset.dataset_names,
        #                                train_dataset_expert_scores =self.train_dataset.expert_scores ,trainer_type = self.trainer_type)
        # else:
        #     self.evaluator = Evaluator(self.model, self.val_dataset, self.val_dataloader, (self.dict_code, self.dict_comment),
        #                           trainer_type = self.trainer_type)

    def train(self,  criterion, optim,  start_time=None):
        

        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time

        scheduler = create_scheduler(optim,
                                     self.config['sl']['warmup_epochs'],
                                     self.config['sl']['warmup_factor'],
                                     self.config['sl']['lr_milestones'],
                                     self.config['sl']['lr_gamma'])

        if not self.distill and self.oriname2finetune  is None :
            expert_outputs = [None for _ in range(len(self.train_dataset))]

            # expert_outputs = [None for _ in range(1,3)] # for debug TODO
            # print("use partial data for debug !!!!!")

        # ###  TODO for debug

        # avg_loss, score_Bleu_1, score_Bleu_2, score_Bleu_3, score_Bleu_4, score_Meteor, score_Rouge, score_Cider, student_scores \
        #     = eval(lm_criterion, cal_meteor=False, epoch=-1, update_best= False )
        # LOGGER.info(
        #     'Epoch: %4d, Val loss: %.5f, Val score_Bleu_1: %.5f Val score_Meteor: %.5f, Val score_Rouge: %.5f, Val score_Cider: %.5f ' % (
        #         -1, avg_loss, score_Bleu_1, score_Meteor, score_Rouge, score_Cider))


        for epoch in range(1,1+self.train_epoch):
            self.model.train()
            iteration, total_loss = 0, 0
            train_data_iter = iter(self.train_dataloader)

            # debug_expert_outputs = [None for _ in range(len(self.train_dataset))] # for debug TODO

            for iteration in range(1,1+ len(self.train_dataloader)):
            # for iteration in range(1,3): # for debug TODO
            #     LOGGER.info("use partial data!!!! for debug ")
                # batch = data_iter.__next__()
                # batch = batch2cuda(self.opt, batch)

                batch = train_data_iter.__next__()
                if self.config['common']['device'] is not None:
                    batch = batch_to_cuda(batch)

                batch_size = batch['tok'][0].shape[0]

                comment_loss,comment_logprobs, comment_target_batch2use = self.model.train_sl_kd(batch, criterion)

                if not self.distill and self.oriname2finetune is None :
                    model_output = torch.exp(comment_logprobs.detach())
                    non_padding_mask = comment_target_batch2use.data.ne(PAD ).cpu()
                    tgtlen = non_padding_mask.shape[-1]
                    topk_idx, topk_v = output2topk(model_output, self.distill_topk)
                    topk_x_shape = (batch_size, tgtlen, self.distill_topk)
                    topk_idx, topk_v = topk_idx.reshape(*topk_x_shape).cpu().numpy(), topk_v.reshape(*topk_x_shape).cpu().numpy()
                    non_padding_mask = non_padding_mask.reshape(*topk_x_shape[:2]).cpu().numpy().astype(bool)
                    for b in range(batch_size):
                        expert_outputs[batch['id'][b]] = \
                            topk_idx[b, non_padding_mask[b]].tolist(), \
                            topk_v[b, non_padding_mask[b]].tolist()
                        # debug_expert_outputs[batch['id'][b]] = sum(non_padding_mask[b])
                        # LOGGER.info("batch['id'][b]:{} sum(non_padding_mask[b]):{} ".format(batch['id'][b] , sum(non_padding_mask[b]) ))



                optim.zero_grad()
                comment_loss.backward()
                total_loss += comment_loss.item()
                # if self.config['flag_clip_grad:
                #     total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm)
                optim.step()


                # if iteration % self.config['training']['log_interval'] == 0 and iteration > 0:
                #     LOGGER.info('Epoch %3d / %3d, %6d/%d batches; avg_loss: %9.5f; %s elapsed' % (
                #         epoch, self.all_epoch, iteration, len(self.train_dataloader), total_loss / iteration,
                #         str(datetime.timedelta(seconds=int(time.time() - self.start_time)))))

                # if iteration % self.config['training']['log_interval'] == 0 and iteration > 0:
                #     LOGGER.info('Epoch: {:0>3d}/{:0>3d}, batches: {:0>3d}/{:0>3d}, avg_loss: {:.8f}; lr: {}, time: {}'.format(
                #         epoch, self.all_epoch, iteration, len(self.train_dataloader),
                #         total_loss / iteration, scheduler.get_lr(),
                #         str(datetime.timedelta(seconds=int(time.time() - start_time)))))

                if iteration % self.config['training']['log_interval'] == 0 and iteration > 0:
                    LOGGER.info('Epoch: {}, batches: {:0>3d}/{:0>3d}, avg_loss: {:.8f}; lr: {}, time: {}'.format(
                        epoch,   iteration, len(self.train_dataloader),
                        total_loss / iteration,  scheduler.get_lr()[0],
                        str(datetime.timedelta(seconds=int(time.time() - self.start_time)))))

            scheduler.step(epoch)

            # Evaluate
            # avg_loss, score_Bleu_1, score_Bleu_2, score_Bleu_3, score_Bleu_4, score_Meteor, score_Rouge, score_Cider ,student_scores\
            #     = eval(criterion, cal_meteor=False,epoch=epoch,update_best = epoch<self.train_epoch)
            # LOGGER.info(
            #     'Epoch: %4d, Val loss: %.5f, Val score_Bleu_1: %.5f Val score_Meteor: %.5f, Val score_Rouge: %.5f, Val score_Cider: %.5f ' % (
            #         epoch, avg_loss, score_Bleu_1, score_Meteor, score_Rouge, score_Cider))

            if self.distill:
                student_scores, self.best_bleu1, self.best_bleu1_epoch ,self.best_cider, self.best_cider_epoch,\
                self.last_best_bleu1_dict\
                    = Evaluator.summarization_kd_eval(args=self.args,
                    last_best_bleu1_dict=self.last_best_bleu1_dict,
                      model=self.model ,dataset=self.val_dataset,data_loader=self.val_dataloader,
                    token_dicts=self.token_dicts, criterion=criterion,trainer_type=self.trainer_type,
                            train_dataset_names = self.train_dataset.dataset_names,
                            train_dataset_expert_scores = self.train_dataset.expert_scores,
                                  epoch=epoch,
                                  best_bleu1=self.best_bleu1,best_bleu1_epoch=self.best_bleu1_epoch,
                                 best_cider = self.best_cider,best_cider_epoch = self.best_cider_epoch,
                                model_filename=None,metrics=['bleu','cider'] )

            else:

                student_scores, self.best_bleu1,  self.best_bleu1_epoch ,self.best_cider, self.best_cider_epoch ,\
                        self.last_best_bleu1_dict \
                    = Evaluator.summarization_kd_eval(args=self.args,
                    last_best_bleu1_dict = self.last_best_bleu1_dict,
                    model=self.model ,dataset=self.val_dataset,data_loader=self.val_dataloader,
                    token_dicts=self.token_dicts, criterion=criterion,trainer_type=self.trainer_type,
                                 epoch=epoch,
                                  best_bleu1=self.best_bleu1,best_bleu1_epoch=self.best_bleu1_epoch,
                    best_cider=self.best_cider, best_cider_epoch=self.best_cider_epoch,
                                  model_filename=None,metrics=['bleu',  'cider'] )


            if self.distill:
                self.train_dataset.student_scores = student_scores



            model_name = \
    '{}-bs{}-lr{}-attn{}-pointer{}-ep{}-tt{}-di{}-slng{}-d{}-hc{}-afs{}-ka{}-dk{}-dp{}-ls{}-kt{}-s{}-o{}-se{}-pr{}-bi{}.pt'.format(
                self.code_modalities_str ,
                self.config['training']['batch_size'],
                self.config['sl']['lr'],
                self.config['training']['attn_type'],
                self.config['training']['pointer'],epoch,self.trainer_type,self.config['kd']['distill'],
                self.config['dataset']['source_domain']['source']['select_lng'][0],self.config['training']['dropout'],
                self.config['training']['enc_hc2dec_hc'],self.config['kd']['alpha_strategy'],
                self.config['kd']['kd_default_alpha'],self.config['kd']['distill_topk'],
                self.config['kd']['distill_temp'],self.config['kd']['label_smooth_rate'],
                self.config['kd']['kd_threshold'],self.config['kd']['shuffle'],self.config['sl']['oriname2finetune'],
                self.source_epoch_str,self.config['dataset']['portion'],self.config['training']['rnn_bidirectional'])



            model_path = os.path.join(self.config['dataset']['save_dir'] ,model_name  )
            torch.save(self.model.state_dict(), model_path)
            LOGGER.info("Save self.model as %s" % model_path)



            if not self.distill and self.oriname2finetune is None :


                path = os.path.join(self.kd_path, '{}_{}_topk_idx_{}_{}_epoch{}'.\
                                    format(self.source[0], self.target[0],
                                           self.trainer_type,self.code_modalities_str,epoch))
                TeacherOutputDataset.save_bin(path, [o[0] for o in expert_outputs], np.int32)
                path = os.path.join(self.kd_path, '{}_{}_topk_prob_{}_{}_epoch{}'.\
                                    format(self.source[0], self.target[0],
                                           self.trainer_type,self.code_modalities_str,epoch))
                TeacherOutputDataset.save_bin(path, [o[1] for o in expert_outputs], np.float)

             #    save_json(debug_expert_outputs,os.path.join(self.config['dataset']['save_dir'] ,
             # 'debug_expert_outputs_epoch{}_{}_{}.json'.format(epoch,self.source[0], self.target[0] )))

             #    path_debug_expert_outputs  = os.path.join(self.config['dataset']['save_dir'] ,
             # 'debug_expert_outputs_epoch{}_{}_{}'.format(epoch,self.source[0], self.target[0] ))
             #    np.save(path_debug_expert_outputs, debug_expert_outputs)

                # if epoch<self.train_epoch and  self.best_bleu1_epoch == epoch : # only save self.model which has best blue1(judged in evaluator)
            # 
            #     LOGGER.info("epoch {} val metric improved ,best_bleu1_epoch: {}   best_bleu1:{} ".\
            #           format(epoch,self.best_bleu1_epoch ,self.best_bleu1))


            # elif epoch>=self.train_epoch:
            #     LOGGER.info("""epoch{} out of self.train_epoch{} ,  not save self.model \n
            #           from epoch 0 to epoch{} best_bleu1_epoch: {}   best_bleu1:{} """.\
            #           format(epoch,self.train_epoch,self.train_epoch, self.best_bleu1_epoch ,self.best_bleu1))
            # elif self.best_bleu1_epoch != epoch and epoch<self.train_epoch:
            #     LOGGER.info(
            #         "epoch{} , val metric not improve , now , best_bleu1_epoch: {}   best_bleu1:{}  ". \
            #         format(epoch, self.best_bleu1_epoch, self.best_bleu1))

            #
            # if epoch == self.train_epoch-1:
            #     LOGGER.info("from epoch 0 to epoch{} best_bleu1_epoch: {}   best_bleu1:{} ".\
            #           format(epoch , self.best_bleu1_epoch ,self.best_bleu1))