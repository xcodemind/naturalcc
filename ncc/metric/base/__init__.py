# -*- coding: utf-8 -*-
from ncc.metric.base.lm_loss import LMLoss
from ncc.metric.base.triple_loss import TripletLoss
from ncc.metric.base.trl_loss import TRLLoss
from ncc.metric.base.retr_loss import RetrievalNLLoss
from ncc.metric.base.labelsmoothing_loss import LMCriterionLabelSmooth,LMCriterionLabelSmoothKD

__all__ = [
    # summarizaiton
    'LMLoss', 'TRLLoss','LMCriterionLabelSmooth','LMCriterionLabelSmoothKD',
    # retrieval
    'RetrievalNLLoss', 'TripletLoss',
]
