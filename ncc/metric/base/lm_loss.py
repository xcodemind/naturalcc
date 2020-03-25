# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from ncc import *
from ncc.metric import *
from ncc.utils.constants import PAD


class _LMLoss(Module):
    '''
    cross-entropy loss
    '''

    __slots__ = ('_gather',)

    def __init__(self, gather=True) -> None:
        super(_LMLoss, self).__init__()
        self._gather = gather

    def forward(self, log_probs: torch.Tensor, target: torch.Tensor, ) -> torch.Tensor:
        '''
        :param log_probs: [batch_size, seq_len, probability_size]
        :param target: [batch_size, seq_len]
        '''
        log_probs = log_probs.reshape(-1, log_probs.size(-1))
        target = target.reshape(-1, 1)
        if self._gather:
            # try:
            #     assert log_probs.size(0) == target.size(0)
            # except Exception as err:
            #     LOGGER.error(err)
            #     LOGGER.error('please increase max_predict_length to {}.'.format(target.size(0)))
            #     assert False
            log_probs_selected = torch.gather(log_probs, 1, target)
        else:
            log_probs_selected = log_probs
        mask = target.data.gt(0)  # generate the mask
        out = torch.masked_select(log_probs_selected, mask)
        loss = -torch.sum(out) / torch.sum(target.data.ne(PAD)).float()  # get the average loss.
        return loss


class LMLoss(BaseLoss):

    def __init__(self, device: bool, gather=True) -> None:
        super(LMLoss, self).__init__(_LMLoss(gather), device, )


if __name__ == '__main__':
    from torch import nn

    m = nn.LogSoftmax(dim=-1)
    batch_size, seq_len, class_num = 3, 5, 10
    input = torch.randn(batch_size, seq_len, class_num, requires_grad=True)
    print(input.size())
    target = torch.randint(0, 10, size=(batch_size, seq_len,))
    print(target.size())

    loss = LMLoss(device=True, gather=True)
    output = loss(m(input), target)
    print(output.item())
    output.backward()
