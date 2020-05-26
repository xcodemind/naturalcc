# -*- coding: utf-8 -*-

import torch

a = torch.Tensor([[1, 0], [0, 1]])
print(a)
b = a.unsqueeze(dim=-1).expand(-1, -1, 2)
print(b)
