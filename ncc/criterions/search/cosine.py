# -*- coding: utf-8 -*-


import math
import torch
import torch.nn.functional as F

from ncc.utils import utils
from ncc.logging import metrics
from ncc.criterions import NccCriterion, register_criterion


@register_criterion('search_cosine')
class SearchSoftmaxCriterion(NccCriterion):
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
        loss, mrr = self.compute_loss(model, net_output, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'ntokens': sample_size,
            'nsentences': sample_size,
            'sample_size': sample_size,
            'mrr': mrr,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, reduce=True):
        src_emb, tgt_emb = net_output  # B x T
        logits = tgt_emb @ src_emb.t()

        lprobs = model.get_normalized_probs(logits, log_probs=True)
        target = torch.arange(logits.size(0)).long().to(logits.device)
        loss = F.nll_loss(
            lprobs,
            target,
            reduction='sum' if reduce else 'none',
        )
        with torch.no_grad():
            correct_scores = logits.diag()
            compared_scores = logits >= correct_scores.unsqueeze(dim=-1)
            mrr = round((1 / compared_scores.sum(dim=-1).float()).mean().item(), 4)
        return loss, mrr

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        metrics.log_scalar('loss', loss_sum / sample_size, sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
