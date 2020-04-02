# -*- coding: utf-8 -*-
import torch
from torch.nn import Module
from ncc.metric import BaseLoss
from typing import Any

class OHEMLoss(BaseLoss):
    '''
    online hard example mining, a mechanism of training method
    +: enhance performance and convergence, valid for data-imbalance
    -: time-consuming
    ref: http://www.erogol.com/online-hard-example-mining-pytorch/
    '''

    __slots__ = ('_ratio',)

    def __init__(self, base: Module, device: bool, ratio=1.0) -> None:
        '''
        :param ratio: 1->all, 0->None
        '''
        super(OHEMLoss, self).__init__(base(reduction='none'), device, )
        self._ratio = ratio

    def forward(self, input: torch.Tensor, target: torch.Tensor, ratio=None) -> Any:
        if ratio is not None:
            self._ratio = ratio

        data_num = input.size(0)
        NUM_LMT = int(self._ratio * data_num)

        raw_loss = self._base(input, target)
        sorted_loss, indices = raw_loss.topk(NUM_LMT)

        return sorted_loss[indices[:NUM_LMT]].mean()


if __name__ == '__main__':
    from torch import nn

    loss = OHEMLoss(nn.NLLLoss, device=True, )
    print(loss)
    # input is of size N x C = 3 x 5
    input = torch.randn(3, 5, requires_grad=True).cuda()
    # each element in target has to have 0 <= value < C
    target = torch.tensor([1, 0, 4]).cuda()
    m = nn.LogSoftmax(dim=1)
    output = loss(m(input), target)
    print(output)
    output.backward()
