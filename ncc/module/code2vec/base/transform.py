# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import Module


class Transform(Module):
    '''
    transform layer: linear -> activate func -> linear
    transform feature to new feature
    '''

    def __init__(self, feature_size: int, act_func=nn.Tanh, ):
        super(Transform, self).__init__()
        self.transform = nn.Sequential(
            nn.Linear(feature_size, feature_size, ),
            act_func(),
            nn.Linear(feature_size, feature_size, ),
        )

    def forward(self, features: torch.Tensor, ) -> torch.Tensor:
        return self.transform(features)
