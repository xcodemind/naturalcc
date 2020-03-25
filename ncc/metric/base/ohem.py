# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

from ncc import *
from ncc.metric import *


class _HashLoss(Module):
    '''
    ref: Deep Cross-Modal Hashing
    ref: https://github.com/WendellGul/DCMH/blob/master/main.py#L127-L132
    '''

    __slots__ = ('_BATCH_ONES', '_CONST_ONES', '_gamma', '_eta',)

    def __init__(self, device: bool, all_size: int, batch_size: int,
                 gamma: float, eta: float, ) -> None:
        super(_HashLoss, self).__init__()

        if device:
            self._BATCH_ONES = torch.ones(batch_size, 1).float().cuda()
            self._CONST_ONES = torch.ones(all_size - batch_size, 1).float().cuda()
        else:
            self._BATCH_ONES = torch.ones(batch_size, 1).float()
            self._CONST_ONES = torch.ones(all_size - batch_size, 1).float()

        self._gamma = gamma
        self._eta = eta

    def forward(self, batch_feature: torch.Tensor, batch_indices: np.array, const_indices: np.array,
                all_features: torch.Tensor, other_all_features: torch.Tensor,
                binary_mat: torch.Tensor, similarity_mat: torch.Tensor, ) -> Any:
        sum_size = batch_feature.size(0) * all_features.size(0)  # batch_size * all_size

        # negative log likelihood loss
        # [?, bit] * [N, bit]^T => [?, N]
        theta = torch.mm(batch_feature, all_features.t()) / 2.0
        nl_loss = -(similarity_mat * theta - torch.log(1.0 + torch.exp(theta)))  # [?, N]

        # quantization loss
        quantization_loss = torch.pow(binary_mat[batch_indices, :] - batch_feature, 2)

        # l2 norm loss
        # [?, bit]^T * [?, 1] => [bit, 1], [N-?, bit]^T * [N-?, 1] => [bit, 1]
        l2norm_loss = torch.pow(torch.mm(batch_feature.t(), self._BATCH_ONES) + \
                                torch.mm(other_all_features[const_indices, :].t(), self._CONST_ONES), 2)

        loss = (nl_loss.sum() + self._gamma * quantization_loss.sum() + self._eta * l2norm_loss.sum()) / sum_size
        return loss


class HashLoss(BaseLoss):

    def __init__(self, device: bool, all_size: int, batch_size: int,
                 gamma: float, eta: float, ) -> None:
        super(HashLoss, self).__init__(
            _HashLoss(device, all_size, batch_size, gamma, eta),
            device, )


if __name__ == '__main__':
    DATA_LEN = 200
    BATCH_SIZE = 32
    BINARY_CODE_BIT = 16

    code_modal = torch.randn(BATCH_SIZE, BINARY_CODE_BIT).float().cuda()
    comment_modal = torch.randn(BATCH_SIZE, BINARY_CODE_BIT).float().cuda()

    code_mat = torch.randn(DATA_LEN, BINARY_CODE_BIT).cuda()  # all data code feature matrix
    comment_mat = torch.randn(DATA_LEN, BINARY_CODE_BIT).cuda()  # all data comment feature matrix

    binary_code = torch.sign(code_mat + comment_mat)


    def calc_similarity(batch_data: torch.Tensor, ALL_DATA: torch.Tensor, ) -> torch.Tensor:
        # calculate batch similarity matrix for batch training
        return (torch.mm(batch_data, ALL_DATA.t()) > 0).float()


    # train code
    batch_indices = np.random.permutation(DATA_LEN)[:BATCH_SIZE]
    const_indices = np.setdiff1d(range(DATA_LEN), batch_indices)
    code_mat[batch_indices, :] = code_modal
    # [?, bit] * [N, bit]^T => [?, N]
    sim_maxtrix = calc_similarity(code_modal, code_mat)

    hash_loss = HashLoss(device=True, all_size=DATA_LEN, batch_size=BATCH_SIZE, gamma=0.1, eta=0.1)

    # loss
    loss = hash_loss(code_modal, batch_indices, const_indices,
                     code_mat, comment_mat, binary_code, sim_maxtrix)
    print(loss.item())