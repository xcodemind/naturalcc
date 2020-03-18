# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.path.abspath('.'))

import torch
import torch.nn.init as init


def init_xavier_linear(linear, std: float, init_bias=True, gain=1, ) -> None:
    # linear with xavier init
    torch.nn.init.xavier_uniform_(linear.weight, gain)
    if init_bias and linear.bias is not None:
        linear.bias.data.normal_(std)


def init_orthogonal_lstm(lstm, std: float, ) -> None:
    # lstm with orthogonal init
    for param in lstm.parameters():
        if len(param.shape) >= 2:
            init.orthogonal_(param.data)
        else:
            init.normal_(param.data, std)


def init_uniform_lstm(lstm, param_init: float, ) -> None:
    # lstm with uniform init
    for p in lstm.parameters():
        p.data.uniform_(-param_init, param_init)


def init_linear_wt(linear, std: float, ) -> None:
    # linear with normal init
    linear.weight.data.normal_(std)
    if linear.bias is not None:
        linear.bias.data.normal_(std)


def init_wt_normal(wt, std: float, ) -> None:
    # weight with normal init
    wt.data.normal_(std)
