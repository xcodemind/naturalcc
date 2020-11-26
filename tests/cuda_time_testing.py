# -*- coding: utf-8 -*-


import torch
from torch import nn


def test_loss_profiling():
    loss = nn.BCEWithLogitsLoss().cuda()
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        input = torch.randn((8, 1, 128, 128)).cuda()
        input.requires_grad = True

        target = torch.randint(1, (8, 1, 128, 128)).cuda().float()

        for i in range(10):
            l = loss(input, target)
            l.backward()
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))


test_loss_profiling()
