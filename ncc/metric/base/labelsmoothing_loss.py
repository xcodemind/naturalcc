# -*- coding: utf-8 -*-
from ncc import LOGGER
import torch
from torch.nn import Module
import torch.nn.functional as F
from ncc.metric import *
from ncc.utils.constants import PAD

class _LMCriterionLabelSmooth(Module):
    __slots__ = ('gather','label_smooth_rate',)
    def __init__(self, gather=True, label_smooth_rate=0.1):
        super(_LMCriterionLabelSmooth, self).__init__()
        self.gather = gather
        self.label_smooth_rate = label_smooth_rate

    def forward(self, lprobs: torch.Tensor, target: torch.Tensor):
        # lprobs: values after log_softmax
        lprobs = lprobs.reshape(-1, lprobs.size(-1))
        target = target.reshape(-1, 1)
        non_pad_mask = target.data.gt(PAD)  # generate the mask
        if self.gather:
            try:
                assert lprobs.shape[0] == target.shape[0]
            except Exception as err:
                LOGGER.error(err)
                LOGGER.error('please increase max_predict_length to {}.'.format(target.shape[0]))
                assert False
            nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        else:
            raise NotImplementedError('gather==False, NotImplementedError')

        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
        eps_i = self.label_smooth_rate / lprobs.size(-1)
        loss = (1. - self.label_smooth_rate) * nll_loss + eps_i * smooth_loss
        loss = loss / torch.sum(non_pad_mask.int()).float()

        return loss

class LMCriterionLabelSmooth(BaseLoss):

    def __init__(self, device: bool, gather=True, label_smooth_rate=0.1) -> None:
        super(LMCriterionLabelSmooth, self).__init__(_LMCriterionLabelSmooth(gather, label_smooth_rate=0.1), device, )


class _LMCriterionLabelSmoothKD(Module):
    __slots__ = ('gather','label_smooth_rate','distill_temp',)
    def __init__(self, gather=True, label_smooth_rate=0.1, distill_temp=1):

        super(_LMCriterionLabelSmoothKD, self).__init__()
        self.gather = gather
        self.label_smooth_rate = label_smooth_rate
        self.distill_temp = distill_temp

    def forward(self, lprobs: torch.Tensor, target: torch.Tensor, batch:dict):
        lprobs = lprobs.reshape(-1, lprobs.size(-1))
        target = target.reshape(-1, 1)
        non_pad_mask = target.data.gt(PAD)  # generate the mask
        if 'alpha' in batch:

            alpha = batch['alpha'].reshape(-1, 1)[non_pad_mask]
        else:
            alpha = 0
        if self.gather:
            nll_prob = -lprobs.gather(dim=-1, index=target)
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        else:
            raise NotImplementedError('gather==False, NotImplementedError')


        eps_i = self.label_smooth_rate / lprobs.size(-1)

        if 'teacher_output' in batch and batch['teacher_output'] is not None and torch.is_tensor(alpha):
            teacher_output = batch['teacher_output']
            net_output_lprobs_t = F.log_softmax(torch.exp(lprobs) / self.distill_temp, -1)
            net_output_lprobs_t = net_output_lprobs_t.reshape(-1, net_output_lprobs_t.shape[-1])
            topk_idx, topk_prob = teacher_output
            topk_idx = topk_idx.reshape(-1, topk_idx.shape[-1])
            topk_prob = topk_prob.reshape(-1, topk_prob.shape[-1])
            topk_prob = F.softmax(topk_prob / self.distill_temp, -1)
            distill_loss = - (net_output_lprobs_t.gather(dim=-1, index=topk_idx) * topk_prob).sum(dim=-1, keepdim=True)

            distill_loss = (distill_loss[non_pad_mask] * alpha).sum()

            nll_loss = (nll_prob[non_pad_mask] * (1 - alpha)).sum()
            smooth_loss = (smooth_loss[non_pad_mask] * (1 - alpha)).sum()
            s_loss = (1. - self.label_smooth_rate) * nll_loss + eps_i * smooth_loss

            loss = distill_loss * self.distill_temp * self.distill_temp + s_loss

        else:
            nll_loss = nll_prob[non_pad_mask].sum()
            smooth_loss = smooth_loss[non_pad_mask].sum()
            s_loss = (1. - self.label_smooth_rate) * nll_loss + eps_i * smooth_loss
            loss = s_loss
        loss = loss / torch.sum(non_pad_mask.int()).float()
        return loss

class LMCriterionLabelSmoothKD(BaseLoss):

    def __init__(self, device: bool, gather=True, label_smooth_rate=0.1, distill_temp=1) -> None:
        super(LMCriterionLabelSmoothKD, self).__init__(_LMCriterionLabelSmoothKD(gather, label_smooth_rate=0.1, distill_temp=1), device, )


### TODO :  _PGCriterionReinforceLabelSmooth    _PGCriterionReinforceLabelSmoothKD

class _PGCriterionReinforceLabelSmooth(Module):
    __slots__ = ('gather','label_smooth_rate',)
    def __init__(self, gather=True, label_smooth_rate=0.1):
        super(_PGCriterionReinforceLabelSmooth, self).__init__()
        self.gather = gather
        self.label_smooth_rate = label_smooth_rate

    def forward(self, lprobs: torch.Tensor, target: torch.Tensor, seq_padding_mask: torch.Tensor, reward: torch.Tensor):
        lprobs = lprobs.reshape(-1, lprobs.size(-1))
        target = target.reshape(-1, 1)
        try:
            non_pad_mask = seq_padding_mask.reshape(-1, 1).bool()
        except:
            non_pad_mask = seq_padding_mask.reshape(-1, 1).byte()
        if self.gather:
            try:
                assert lprobs.shape[0] == target.shape[0]
            except Exception as err:
                LOGGER.error(err)
                LOGGER.error('please increase max_predict_length to {}.'.format(target.shape[0]))
                assert False
            assert lprobs.shape[0] == target.shape[0], print("please increase opt.max_predict_length")
            nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        else:
            # nll_loss = -lprobs[non_pad_mask]
            # smooth_loss = -lprob_sum.reshape(-1, 1)[non_pad_mask]
            raise NotImplementedError('gather==False, NotImplementedError')

        eps_i = self.label_smooth_rate / lprobs.size(-1)
        loss = (1. - self.label_smooth_rate) * nll_loss + eps_i * smooth_loss
        reward = torch.masked_select(reward.reshape(-1, 1), non_pad_mask)
        loss = torch.sum(loss * reward) / torch.sum(non_pad_mask.int()).float()

        return loss

class PGCriterionReinforceLabelSmooth(BaseLoss):

    def __init__(self, device: bool, gather=True) -> None:
        super(PGCriterionReinforceLabelSmooth, self).__init__(_PGCriterionReinforceLabelSmooth(gather), device, )



class _PGCriterionReinforceLabelSmoothKD(Module):
    __slots__ = ('gather','label_smooth_rate','distill_temp',)
    def __init__(self, gather=True, label_smooth_rate=0.1, distill_temp=1):
        super(_PGCriterionReinforceLabelSmoothKD, self).__init__()
        self.gather = gather
        self.label_smooth_rate = label_smooth_rate
        self.distill_temp = distill_temp

    def forward(self, lprobs, target, seq_padding_mask, reward,  batch):
        lprobs = lprobs.reshape(-1, lprobs.size(-1))
        target = target.reshape(-1, 1)
        try:
            non_pad_mask = seq_padding_mask.reshape(-1, 1).bool()
        except:
            non_pad_mask = seq_padding_mask.reshape(-1, 1).byte()
        if 'alpha' in batch:
            alpha = batch['alpha'].reshape(-1, 1)[non_pad_mask]
        else:
            alpha = 0
        if self.gather:
            nll_prob = -lprobs.gather(dim=-1, index=target)
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        else:
            raise NotImplementedError('gather==False, NotImplementedError')

        eps_i = self.label_smooth_rate / lprobs.size(-1)
        reward = torch.masked_select(reward.reshape(-1, 1), non_pad_mask)
        if 'teacher_output' in batch and batch['teacher_output'] is not None and torch.is_tensor(alpha):

            net_output_lprobs_t = F.log_softmax(torch.exp(lprobs) / self.distill_temp, -1)
            net_output_lprobs_t = net_output_lprobs_t.reshape(-1, net_output_lprobs_t.shape[-1])
            teacher_output = batch['teacher_output']
            topk_idx, topk_prob = teacher_output
            topk_idx = topk_idx.reshape(-1, topk_idx.shape[-1])
            # TODO verify：  reshape  最后一个维度是 topk_idx.shape[-1] ，然后topk_idx就给net_output_lprobs_t来gather了？
            non_pad_mask_dataset = topk_idx.reshape(-1, 1).data.gt(PAD)
            topk_prob = topk_prob.reshape(-1, topk_prob.shape[-1])
            topk_prob = F.softmax(topk_prob / self.distill_temp, -1)
            distill_loss = - (net_output_lprobs_t.gather(dim=-1, index=topk_idx) * topk_prob).sum(dim=-1, keepdim=True)
            distill_loss = (distill_loss[non_pad_mask_dataset] * alpha)
            nll_loss = (nll_prob[non_pad_mask] * (1 - alpha))
            smooth_loss = (smooth_loss[non_pad_mask] * (1 - alpha))
            s_loss = ((  1. - self.label_smooth_rate) * nll_loss + eps_i * smooth_loss) * reward
            s_loss = torch.sum(s_loss) / torch.sum(non_pad_mask.int()).float()
            distill_loss = torch.sum(distill_loss * self.distill_temp * self.distill_temp) / torch.sum(
                non_pad_mask_dataset.int()).float()
            loss = distill_loss + s_loss
        else:
            nll_loss = nll_prob[non_pad_mask].sum()
            smooth_loss = smooth_loss[non_pad_mask].sum()
            loss = torch.sum((1. - self.label_smooth_rate) * nll_loss + eps_i * smooth_loss)
            loss = loss / torch.sum(non_pad_mask.int()).float()

        return loss


class PGCriterionReinforceLabelSmoothKD(BaseLoss):

    def __init__(self, device: bool, gather=True) -> None:
        super(PGCriterionReinforceLabelSmoothKD, self).__init__(_PGCriterionReinforceLabelSmoothKD(gather), device, )