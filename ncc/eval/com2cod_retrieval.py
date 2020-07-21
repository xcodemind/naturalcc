# -*- coding: utf-8 -*-

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import sys

from ncc.utils import utils
import numpy as np


class Com2CodeRetrievalScorer(object):
    """
    comment-to-code retrieval task
    Scores the target for a given source sentence.
    """

    def __init__(self, tgt_dict, softmax_batch=None, compute_alignment=False):
        self.pad = tgt_dict.pad()
        self.softmax_batch = softmax_batch or sys.maxsize
        assert self.softmax_batch > 0
        self.compute_alignment = compute_alignment

    @torch.no_grad()
    def compute(self, models, sample, predict_type, **kwargs):
        """Score a batch of retrieval."""
        net_input = sample['net_input']

        def _mrr_score(similarity_scores):
            # extract the logits from the diagonal of the matrix, which are the logits corresponding to the ground-truth
            correct_scores = torch.diag(similarity_scores)
            # compute how many queries have bigger logits than the ground truth (the diagonal) -> which will be incorrectly ranked
            compared_scores = similarity_scores >= correct_scores.unsqueeze(dim=-1)
            # for each row of the matrix (query), sum how many logits are larger than the ground truth
            # ...then take the reciprocal of that to get the MRR for each individual query (you will need to take the mean later)
            mrr = 1 / compared_scores.sum(dim=1).float()
            return mrr.numpy()

        def _acc_score(similarity_scores):
            rank = torch.argmax(similarity_scores, dim=-1)
            bsz = rank.numel()
            acc = (rank == torch.arange(bsz).to(rank.device)).int()
            return acc.numpy()

        hypos = []
        for model in models:
            model.eval()
            code_repr, query_repr = model(**net_input)
            similarity_scores = query_repr @ code_repr.t()
            mrr = _mrr_score(similarity_scores)
            acc = _acc_score(similarity_scores)

            for _mmr_sc, _acc_sc in zip(mrr, acc):
                hypos.append({
                    'accuracy': _acc_sc,
                    'mrr': _mmr_sc,
                })

        if len(models) > 1:
            ...

        return hypos
