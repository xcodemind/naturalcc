# -*- coding: utf-8 -*-


from ncc.criterions import FairseqCriterion, register_criterion

import torch
import torch.nn.functional as F


@register_criterion('search_softmax')
class SearchSoftmaxCriterion(FairseqCriterion):
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
        loss, _ = self.compute_loss(model, net_output, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'ntokens': sample_size,
            'nsentences': sample_size,
            'sample_size': sample_size,
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
        return loss, loss
