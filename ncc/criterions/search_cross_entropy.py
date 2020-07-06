# -*- coding: utf-8 -*-


from .cross_entropy import CrossEntropyCriterion
from ncc.criterions import register_criterion

import torch
import torch.nn.functional as F


@register_criterion('search_cross_entropy')
class SearchCrossEntropyCriterion(CrossEntropyCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task, sentence_avg)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            # 'ntokens': sample_size,
            # 'nsentences': sample_size,
            # 'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        src_emb, tgt_emb = net_output  # B x T
        src_emb = model.get_normalized_probs(src_emb, log_probs=True)
        tgt_emb = model.get_normalized_probs(tgt_emb, log_probs=True)
        logits = src_emb @ tgt_emb.t()
        target = torch.arange(logits.size(0)).long().to(logits.device)

        loss = F.cross_entropy(
            logits,
            target,
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )
        return loss
