# -*- coding: utf-8 -*-
import sys

sys.path.append('./')

from ncc import *
from ncc.metric import *


class BaseLoss(Module):
    __slots__ = ('_base', '_device',)

    def __init__(self, base: Module, device: bool, ) -> None:
        '''
        :param base: loss base
        :param device: False -> 'CPU' | True -> 'GPU'
        :param reduction(string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of elements in the output,
            ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

        '''
        super(BaseLoss, self).__init__()
        self._base = base
        self._device = 'GPU' if device else 'CPU'
        if self._device == 'CPU':
            pass
        elif self._device == 'GPU':
            self._base.to(torch.cuda.current_device())
        else:
            try:
                raise NotImplementedError('Python interpreter fails. pls, try again.')
            except Exception as err:
                LOGGER.error(err)
                assert False
        LOGGER.debug('building {}'.format(str(self)))

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._base.forward(*args, **kwargs)

    def __str__(self) -> str:
        if hasattr(self._base, 'reduction'):
            return '{}(\'{}\')-{}'.format(str(self._base)[:-2], self._base.reduction, self._device, )
        else:
            return '{}-{}'.format(str(self._base), self._device, )


if __name__ == '__main__':
    loss = BaseLoss(torch.nn.NLLLoss(), device=True, )
    print(loss)
    # input is of size N x C = 3 x 5
    input = torch.randn(3, 5, requires_grad=True).cuda()
    # each element in target has to have 0 <= value < C
    target = torch.tensor([1, 0, 4]).cuda()
    m = torch.nn.LogSoftmax(dim=1)
    output = loss(m(input), target)
    print(output.item())
    output.backward()
