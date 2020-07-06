# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
import numpy as np
# from fairseq import metrics, utils
from ncc.logging import metrics
from ncc.utils import utils
from ncc.criterions import FairseqCriterion, register_criterion


@register_criterion('completion_cross_entropy')
class CompletionCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, ncorrect, mrr = self.compute_loss(model, net_output, sample, reduce=reduce)

        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        loss /= sample['loss_mask'].sum()

        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'ncorrect': ncorrect,
            'mrr': mrr,
        }

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        loss_mask = sample['loss_mask']
        lprobs = lprobs.view(-1, lprobs.size(-1))
        lprobs = lprobs[loss_mask].contiguous()
        target = model.get_targets(sample, net_output).view(-1)
        target = target[loss_mask].contiguous()
        loss = F.nll_loss(
            lprobs,
            target,
            # ignore_index=self.padding_idx,  # skip this line, because loss_mask has skipped those padding idx
            reduction='sum' if reduce else 'none',
        )
        rank = torch.argmax(lprobs, 1)
        mrr = np.mean([1. / (r.item() + 1) for r in rank.view(-1)])

        ncorrect = torch.sum(rank == target)
        return loss, ncorrect, mrr

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
        mrr = sum(log.get('mrr', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('accuracy', ncorrect / ntokens / math.log(2), sample_size, round=3)  # why log2
        metrics.log_scalar('mrr', mrr / sample_size / math.log(2), sample_size, round=3)

        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        else:
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
