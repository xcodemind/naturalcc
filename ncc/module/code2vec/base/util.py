# -*- coding: utf-8 -*-
import torch
from typing import Any

def pooling1d(input_emb: torch.Tensor, input_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == 'mean':
        input_emb = input_emb.sum(dim=1) / input_mask.sum(-1, keepdim=True)
    elif pooling == 'max':
        input_emb, _ = input_emb.max(dim=1)
    else:
        raise NotImplementedError('No such pooling method, only [mean/max] pooling are available')
    return input_emb


def pad_conv1d(raw_seq: torch.Tensor, left: int, right: int) -> torch.Tensor:
    if left != 0 and right == 0:
        return torch.cat(
            [torch.zeros(*raw_seq.size()[:-1], left, dtype=raw_seq.dtype, device=raw_seq.device), raw_seq, ], dim=-1)
    elif left == 0 and right != 0:
        return torch.cat(
            [raw_seq, torch.zeros(*raw_seq.size()[:-1], right, dtype=raw_seq.dtype, device=raw_seq.device), ], dim=-1)
    else:
        return torch.cat(
            [torch.zeros(*raw_seq.size()[:-1], left, dtype=raw_seq.dtype, device=raw_seq.device), raw_seq,
             torch.zeros(*raw_seq.size()[:-1], right, dtype=raw_seq.dtype, device=raw_seq.device), ],
            dim=-1)


def conate_tensor_tuple(tensor: Any) -> torch.Tensor:
    while type(tensor) in [list, tuple]:
        tensor = torch.cat(tensor, dim=-1)
    return tensor


__all__ = [
    'pooling1d', 'pad_conv1d', 'conate_tensor_tuple'
]
