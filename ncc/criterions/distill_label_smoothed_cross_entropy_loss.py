# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
import torch
import torch.nn.functional as F
from ncc.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from ncc.utils import utils
from ncc.criterions import register_criterion


@register_criterion('distill_label_smoothed_cross_entropy')
class DistillLabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, task, sentence_avg, label_smoothing=0.1, distill_temp=0.6):
        super().__init__(task, sentence_avg, label_smoothing)
        self.t = distill_temp

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        net_output = net_output[0].float()
        lprobs = F.log_softmax(net_output, -1)
        lprobs = lprobs.view(-1, lprobs.shape[-1])
        target = sample['target'].view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        if 'alpha' in sample:
            alpha = sample['alpha'].view(-1, 1)[non_pad_mask]
        else:
            alpha = 0

        nll_prob = -lprobs.gather(dim=-1, index=target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']

        eps_i = self.eps / lprobs.size(-1)

        if 'teacher_output' in sample and sample['teacher_output'] is not None and torch.is_tensor(alpha):
            teacher_output = sample['teacher_output']
            net_output_lprobs_t = F.log_softmax(net_output / self.t, -1)
            net_output_lprobs_t = net_output_lprobs_t.view(-1, net_output_lprobs_t.shape[-1])

            topk_idx, topk_prob = teacher_output
            topk_idx = topk_idx.view(-1, topk_idx.shape[-1])
            topk_prob = topk_prob.view(-1, topk_prob.shape[-1])

            topk_prob = F.softmax(topk_prob / self.t, -1)

            distill_loss = - (net_output_lprobs_t.gather(dim=-1, index=topk_idx) * topk_prob).sum(dim=-1,
                                                                                                  keepdim=True)
            distill_loss = (distill_loss[non_pad_mask] * alpha).sum()

            nll_loss = (nll_prob[non_pad_mask] * (1 - alpha)).sum()
            smooth_loss = (smooth_loss[non_pad_mask] * (1 - alpha)).sum()
            s_loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss

            loss = distill_loss * self.t * self.t + s_loss
            nll_loss = nll_prob[non_pad_mask].sum()
        else:
            nll_loss = nll_prob[non_pad_mask].sum()
            smooth_loss = smooth_loss[non_pad_mask].sum()
            s_loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
            loss = s_loss

        logging_output = {
            'loss': utils.item(loss.item()) if reduce else loss.item(),
            'nll_loss': utils.item(nll_loss.item()) if reduce else nll_loss.item(),
            'nsentences': sample['target'].size(0),
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
        }

        return loss, sample_size, logging_output
