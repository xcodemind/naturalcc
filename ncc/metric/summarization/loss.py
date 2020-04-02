# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from ncc.utils.constants import *

# from ncc.log.log import get_logger

# LOGGER = get_logger()


# class LMCriterion(nn.Module):
#     def __init__(self, gather=True) -> None:
#         super(LMCriterion, self).__init__()
#         self.gather = gather
#
#     def forward(self, lprobs, target):
#         '''
#         :param lprobs: [batch_size, seq_len, probability_size]
#         :param target: [batch_size, seq_len]
#         '''
#         lprobs = lprobs.reshape(-1, lprobs.size(-1))
#         target = target.reshape(-1, 1)
#         if self.gather:
#             try:
#                 assert lprobs.size(0) == target.size(0)
#             except Exception as err:
#                 LOGGER.error(err)
#                 LOGGER.error('please increase max_predict_length to {}.'.format(target.size(0)))
#                 assert False
#             logprob_select = torch.gather(lprobs, 1, target)
#         else:
#             logprob_select = lprobs
#         mask = target.gt(0)  # generate the mask .data
#         out = torch.masked_select(logprob_select, mask)
#         loss = -torch.sum(out) / torch.sum(target.ne(PAD)).float()  # get the average loss. .data
#         return loss


class PGCriterion_REINFORCE(nn.Module):
    def __init__(self, gather=True):
        super(PGCriterion_REINFORCE, self).__init__()
        self.gather = gather

    def forward(self, lprobs, target, seq_padding_mask, reward):
        lprobs = lprobs.reshape(-1, lprobs.size(-1))
        target = target.reshape(-1, 1)
        if self.gather:
            logprob_select = torch.gather(lprobs, 1, target)
        else:
            logprob_select = lprobs
        try:
            mask = seq_padding_mask.reshape(-1, 1).bool()
        except:
            mask = seq_padding_mask.reshape(-1, 1).byte()
        out = torch.masked_select(logprob_select, mask)
        reward = torch.masked_select(reward.reshape(-1, 1), mask)
        out = out * reward
        # loss = -torch.sum(out)   # get the average loss.
        loss = -torch.sum(out) / torch.sum(seq_padding_mask).float()  # get the average loss.
        return loss


# class LMCriterionLabelSmooth(nn.Module):
#     def __init__(self, gather=True, label_smooth_rate=0.1):
#         super(LMCriterionLabelSmooth, self).__init__()
#         self.gather = gather
#         self.label_smooth_rate = label_smooth_rate
#
#     def forward(self, lprobs, target, lprob_sum=None):
#         # lprobs: values after log_softmax
#         lprobs = lprobs.reshape(-1, lprobs.size(-1))
#         target = target.reshape(-1, 1)
#         non_pad_mask = target.data.gt(PAD)  # generate the mask
#         if self.gather:
#             assert lprobs.shape[0] == target.shape[0], print(
#                 "please increase opt.max_predict_length , lprobs.shape:{} target.shape:{}". \
#                     format(lprobs.shape, target.shape))
#             # logprob_select = torch.gather(lprobs, 1, target)
#             nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
#             smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
#         else:
#             nll_loss = -lprobs[non_pad_mask]
#             smooth_loss = -lprob_sum.reshape(-1, 1)[non_pad_mask]
#         # out = torch.masked_select(logprob_select, mask)
#         # loss = -torch.sum(out)  # get the average loss.
#         nll_loss = nll_loss.sum()
#         # print("======\nbefore sum , smooth_loss: ", smooth_loss)
#         smooth_loss = smooth_loss.sum()
#         # print("torch.sum(non_pad_mask.float()): ",torch.sum(non_pad_mask.float()) )
#         # print("after sum , smooth_loss: ",smooth_loss )
#         eps_i = self.label_smooth_rate / lprobs.size(-1)
#         loss = (1. - self.label_smooth_rate) * nll_loss + eps_i * smooth_loss
#         loss = loss / torch.sum(non_pad_mask.int()).float()
#         # print("LMCriterionLabelSmooth  torch.sum(non_pad_mask.int()).float()  : ", torch.sum(non_pad_mask.int()).float() )
#         # print("LMCriterionLabelSmooth  non_pad_mask : ", non_pad_mask  )
#         # print("LMCriterionLabelSmooth  non_pad_mask.int() : ", non_pad_mask.int()    )
#         return loss
#
#
# class LMCriterionLabelSmoothKD(nn.Module):
#     # def __init__(self,gather=True,label_smooth_rate=0.1,distill_temp=1):
#     def __init__(self, gather=True, label_smooth_rate=0.1, distill_temp=1):
#
#         super(LMCriterionLabelSmoothKD, self).__init__()
#         self.gather = gather
#         self.label_smooth_rate = label_smooth_rate
#         self.distill_temp = distill_temp
#
#     def forward(self, lprobs, target, lprob_sum, batch):
#         lprobs = lprobs.reshape(-1, lprobs.size(-1))
#         target = target.reshape(-1, 1)
#         non_pad_mask = target.data.gt(PAD)  # generate the mask
#         if 'alpha' in batch:
#             # print("non_pad_mask.shape: ",non_pad_mask.shape )
#             # print("batch['alpha'].shape: ",batch['alpha'].shape )
#             alpha = batch['alpha'].reshape(-1, 1)[non_pad_mask]
#         else:
#             alpha = 0
#         if self.gather:
#             nll_prob = -lprobs.gather(dim=-1, index=target)
#             smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
#         else:
#             assert False, print("need lprobs before gather!!!!")
#             # lprob_sum = lprob_sum.reshape(-1, lprob_sum.shape[-1])
#             # nll_prob = -lprobs
#             # smooth_loss = -lprob_sum
#
#         eps_i = self.label_smooth_rate / lprobs.size(-1)
#
#         if 'teacher_output' in batch and batch['teacher_output'] is not None and torch.is_tensor(alpha):
#             # opt.distill = 1 , train , enter this branch
#             teacher_output = batch['teacher_output']
#             net_output_lprobs_t = F.log_softmax(torch.exp(lprobs) / self.distill_temp, -1)
#             # print("==========================")
#             # print("net_output_lprobs_t.shape: ",net_output_lprobs_t.shape )
#             net_output_lprobs_t = net_output_lprobs_t.reshape(-1, net_output_lprobs_t.shape[-1])
#             topk_idx, topk_prob = teacher_output
#             # print("topk_idx.shape: ", topk_idx.shape)
#             # print("topk_prob.shape: ", topk_prob.shape)
#             topk_idx = topk_idx.reshape(-1, topk_idx.shape[-1])
#             topk_prob = topk_prob.reshape(-1, topk_prob.shape[-1])
#             topk_prob = F.softmax(topk_prob / self.distill_temp, -1)
#             # print("=====")
#             # print("net_output_lprobs_t.shape: ",net_output_lprobs_t.shape )
#             # print("topk_idx.shape: ",topk_idx.shape )
#             # # print("topk_idx: ",topk_idx)
#             # print("topk_prob.shape: ",topk_prob.shape)
#             distill_loss = - (net_output_lprobs_t.gather(dim=-1, index=topk_idx) * topk_prob).sum(dim=-1, keepdim=True)
#             # distill_loss = - (lprobs.gather(dim=-1, index=topk_idx) * topk_prob).sum(dim=-1,keepdim=True)
#             distill_loss = (distill_loss[non_pad_mask] * alpha).sum()
#
#             nll_loss = (nll_prob[non_pad_mask] * (1 - alpha)).sum()
#             smooth_loss = (smooth_loss[non_pad_mask] * (1 - alpha)).sum()
#             s_loss = (1. - self.label_smooth_rate) * nll_loss + eps_i * smooth_loss
#
#             # loss = distill_loss * self.t * self.t + s_loss
#             loss = distill_loss * self.distill_temp * self.distill_temp + s_loss
#             # nll_loss = nll_prob[non_pad_mask].sum()
#         else:  # opt.distill = 1 , eval and test , enter this branch
#             # assert False, print('no tearcher output in batch!!')
#             nll_loss = nll_prob[non_pad_mask].sum()
#             smooth_loss = smooth_loss[non_pad_mask].sum()
#             s_loss = (1. - self.label_smooth_rate) * nll_loss + eps_i * smooth_loss
#             loss = s_loss
#         loss = loss / torch.sum(non_pad_mask.int()).float()
#         return loss
#
#
# class PGCriterionReinforceLabelSmooth(nn.Module):
#     def __init__(self, gather=True, label_smooth_rate=0.1):
#         super(PGCriterionReinforceLabelSmooth, self).__init__()
#         self.gather = gather
#         self.label_smooth_rate = label_smooth_rate
#
#     def forward(self, lprobs, target, seq_padding_mask, reward, lprob_sum=None):
#         lprobs = lprobs.reshape(-1, lprobs.size(-1))
#         target = target.reshape(-1, 1)
#         try:
#             non_pad_mask = seq_padding_mask.reshape(-1, 1).bool()
#         except:
#             non_pad_mask = seq_padding_mask.reshape(-1, 1).byte()
#         # target_mask = target.data.gt(PAD )  # generate the mask
#         # assert non_pad_mask == target_mask, print("naturalcode/src/metric/codesum/loss.py line95 , non_pad_mask:\n{}  target_mask:\n{}".format(non_pad_mask,target_mask))
#         if self.gather:
#             assert lprobs.shape[0] == target.shape[0], print("please increase opt.max_predict_length")
#             nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
#             smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
#         else:
#             # lprob_sum = lprob_sum.reshape(-1, lprob_sum.shape[-1])
#             nll_loss = -lprobs[non_pad_mask]
#             smooth_loss = -lprob_sum.reshape(-1, 1)[non_pad_mask]
#
#         eps_i = self.label_smooth_rate / lprobs.size(-1)
#         loss = (1. - self.label_smooth_rate) * nll_loss + eps_i * smooth_loss
#
#         reward = torch.masked_select(reward.reshape(-1, 1), non_pad_mask)
#         # loss = torch.sum(loss * reward)  # get the average loss.
#         loss = torch.sum(loss * reward) / torch.sum(non_pad_mask.int()).float()
#
#         return loss
#
#
# class PGCriterionReinforceLabelSmoothKD(nn.Module):
#     def __init__(self, gather=True, label_smooth_rate=0.1, distill_temp=1):
#         super(PGCriterionReinforceLabelSmoothKD, self).__init__()
#         self.gather = gather
#         self.label_smooth_rate = label_smooth_rate
#         self.distill_temp = distill_temp
#
#     def forward(self, lprobs, target, seq_padding_mask, reward, lprob_sum, batch):
#         lprobs = lprobs.reshape(-1, lprobs.size(-1))
#         target = target.reshape(-1, 1)
#         # non_pad_mask = target.data.gt(PAD)  # generate the mask
#         try:
#             non_pad_mask = seq_padding_mask.reshape(-1, 1).bool()
#         except:
#             non_pad_mask = seq_padding_mask.reshape(-1, 1).byte()
#         if 'alpha' in batch:
#             alpha = batch['alpha'].reshape(-1, 1)[non_pad_mask]
#         else:
#             alpha = 0
#         if self.gather:
#             nll_prob = -lprobs.gather(dim=-1, index=target)
#             smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
#         else:
#             assert False, print("need lprobs before gather")
#             # lprob_sum = lprob_sum.reshape(-1,lprob_sum.shape[-1])
#             # nll_prob = -lprobs
#             # smooth_loss = -lprob_sum
#         eps_i = self.label_smooth_rate / lprobs.size(-1)
#         reward = torch.masked_select(reward.reshape(-1, 1), non_pad_mask)
#         if 'teacher_output' in batch and batch['teacher_output'] is not None and torch.is_tensor(alpha):
#
#             net_output_lprobs_t = F.log_softmax(torch.exp(lprobs) / self.distill_temp, -1)
#             net_output_lprobs_t = net_output_lprobs_t.reshape(-1, net_output_lprobs_t.shape[-1])
#             teacher_output = batch['teacher_output']
#             topk_idx, topk_prob = teacher_output
#             topk_idx = topk_idx.reshape(-1, topk_idx.shape[-1])
#             # TODO verify：  reshape  最后一个维度是 topk_idx.shape[-1] ，然后topk_idx就给net_output_lprobs_t来gather了？
#             non_pad_mask_dataset = topk_idx.reshape(-1, 1).data.gt(PAD)
#             topk_prob = topk_prob.reshape(-1, topk_prob.shape[-1])
#             topk_prob = F.softmax(topk_prob / self.distill_temp, -1)
#             distill_loss = - (net_output_lprobs_t.gather(dim=-1, index=topk_idx) * topk_prob).sum(dim=-1, keepdim=True)
#             # distill_loss = - (lprobs.gather(dim=-1, index=topk_idx) * topk_prob).sum(dim=-1,keepdim=True)
#             # distill_loss = (distill_loss[non_pad_mask] * alpha).sum()
#             distill_loss = (distill_loss[non_pad_mask_dataset] * alpha)
#             # nll_loss = (nll_prob[non_pad_mask] * (1 - alpha)).sum()
#             nll_loss = (nll_prob[non_pad_mask] * (1 - alpha))
#             # smooth_loss = (smooth_loss[non_pad_mask] * (1 - alpha)).sum()
#             smooth_loss = (smooth_loss[non_pad_mask] * (1 - alpha))
#             s_loss = ((
#                               1. - self.label_smooth_rate) * nll_loss + eps_i * smooth_loss) * reward  # not consider reward in distill_loss
#             s_loss = torch.sum(s_loss) / torch.sum(non_pad_mask.int()).float()
#             # loss = torch.sum(distill_loss*self.distill_temp*self.distill_temp  + s_loss )
#             distill_loss = torch.sum(distill_loss * self.distill_temp * self.distill_temp) / torch.sum(
#                 non_pad_mask_dataset.int()).float()
#             loss = distill_loss + s_loss
#             # reason that we multiply distill_loss by self.distill_temp twice : 《Distilling the Knowledge in a Neural Network》
#         else:
#             # assert False, print('no tearcher output in batch!!')
#             nll_loss = nll_prob[non_pad_mask].sum()
#             smooth_loss = smooth_loss[non_pad_mask].sum()
#             loss = torch.sum((1. - self.label_smooth_rate) * nll_loss + eps_i * smooth_loss)
#             loss = loss / torch.sum(non_pad_mask.int()).float()
#
#         return loss
#
#
# # class PGCriterion_REINFORCE_no_gather(nn.Module):
# #     def __init__(self):
# #         super(PGCriterion_REINFORCE_no_gather, self).__init__()
# #
# #     def forward(self, lprobs, target, seq_padding_mask, reward):
# #         # logprob_select = torch.gather(lprobs, 1, target)
# #         mask = seq_padding_mask.reshape(-1, 1).byte()
# #         out = torch.masked_select(lprobs, mask)
# #         reward = torch.masked_select(reward.reshape(-1, 1), mask)
# #         out = out * reward
# #         loss = -torch.sum(out)  # get the average loss.
# #         return loss


class TRLCriterion(nn.Module):
    '''
    reference: Deep transfer reinforcement learning for text summarization
    CE loss= cross_entropy() # target data, eg. c#
    TRL loss = -1 * sum(
                        (1 - eta) * log(greedy_probability) * (greedy_reward - sampled_reward) # target data, eg.c#
                        - eta * log(greedy_probability) * (greedy_reward - sampled_reward) # source data, eg.c
                        )
    '''

    def __init__(self):
        super(TRLCriterion, self).__init__()

    def CE_Loss(self, log_probs, target):
        '''
        :param lprobs: [batch_size, seq_len, probability_size]
        :param target: [batch_size, seq_len]
        copy from LMCriterion
        '''
        log_probs = log_probs.reshape(-1, log_probs.size(-1))
        target = target.reshape(-1, 1)
        log_probs_select = torch.gather(log_probs, 1, target)
        mask = target.data.gt(0)  # generate the mask
        out = torch.masked_select(log_probs_select, mask)
        ce_loss = -torch.sum(out) / torch.sum(target.data.ne(PAD)).float()  # get the average loss.
        return ce_loss

    def RL_Loss(self, log_probs, target, reward_diff):
        log_probs_select = torch.gather(log_probs, -1, target.unsqueeze(-1)).squeeze(-1)
        mask = target.data.gt(0)  # generate the mask
        out = mask.float() * -log_probs_select * reward_diff
        rl_loss = (out.sum(-1) / target.data.ne(PAD).sum(-1).float()).mean()
        return rl_loss

    def forward(self, log_probs, target, reward_diff, eta=0.1, zeta=0.1):
        '''
        '''
        if eta == 0.0:
            ce_loss = self.CE_Loss(log_probs, target)  # crossy entropy for all
            return ce_loss
        else:
            batch_size = log_probs.size(0) // 2
            # log probability
            src_log_probs, trg_log_probs = log_probs[:batch_size, ...], log_probs[batch_size:, ...]
            # target
            src_target, trg_target = target[:batch_size, ...], target[batch_size:, ...]
            # reward difference
            src_reward_diff, trg_reward_diff = reward_diff[:batch_size, ...], reward_diff[batch_size:, ...]

            trg_ce_loss = self.CE_Loss(trg_log_probs, trg_target)
            src_rl_loss = self.RL_Loss(src_log_probs, src_target, src_reward_diff)  # source, eg. c, java, python
            trg_rl_loss = self.RL_Loss(trg_log_probs, trg_target, trg_reward_diff)  # target, eg c#

            return (1.0 - eta) * trg_ce_loss + eta * (
                    (1.0 - zeta) * trg_rl_loss + zeta * src_rl_loss
            )
