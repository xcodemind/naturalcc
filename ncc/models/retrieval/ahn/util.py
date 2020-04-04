# -*- coding: utf-8 -*-
import torch

def calc_sim_mat(batch_input1: torch.Tensor, batch_input2: torch.Tensor, ) -> torch.Tensor:
    # calculate the similar matrix
    sim_mat = (torch.mm(batch_input1, batch_input2.t()) > 0).to(batch_input1.device).float()
    return sim_mat


def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)


def one_hot_encode(labels: torch.Tensor, class_num: int) -> torch.Tensor:
    one_hot = torch.zeros(labels.size(0), class_num).byte().to(labels.device)
    one_hot.scatter_(1, labels.view(-1, 1), 1)
    return one_hot


def hamming_dist(b_code1: torch.Tensor, b_code2: torch.Tensor, ):
    return (b_code1 != b_code2).sum(dim=-1).float().mean()


if __name__ == '__main__':
    bs = 2
    code_len = 5
    code1 = torch.randn(bs, code_len) > 0.5
    print(code1)
    code2 = torch.randn(bs, code_len) > 0.5
    print(code2)
    dist = hamming_dist(code1, code2)
    print(dist)
