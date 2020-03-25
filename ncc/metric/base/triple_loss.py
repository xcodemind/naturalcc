# -*- coding: utf-8 -*-
import sys

sys.path.append('./')

from ncc import *
from ncc.metric import *
from torch.nn import MarginRankingLoss, SoftMarginLoss
from torch.autograd import Variable


def euclidean_dist(tensor1: torch.Tensor, tensor2: torch.Tensor, ) -> torch.Tensor:
    # tensor euclidean distance
    bs1, bs2 = tensor1.size(0), tensor2.size(0)
    tensor1_squared = torch.pow(tensor1, 2).sum(1, keepdim=True).expand(bs1, bs2)
    tensor2_squared = torch.pow(tensor2, 2).sum(1, keepdim=True).expand(bs2, bs1).t()
    # dist = x^2 + y^2
    eucl_dist = tensor1_squared + tensor2_squared
    # dist - 2 * x * yT
    eucl_dist.addmm_(1, -2, tensor1, tensor2.t())
    eucl_dist = eucl_dist.clamp(min=1e-12).sqrt()
    return eucl_dist


class TripletLoss(BaseLoss):
    __slots__ = ('_margin', '_mining',)

    def __init__(self, device: bool, margin=None, mining=False, ) -> None:
        self._margin = margin  # margin should be small
        self._mining = mining
        if margin is not None:
            '''
            MarginRankingLoss(x1, x2, y) = max(0, -y*(x1-x2) + margin)
            if y=1
                max(0, -x_neg + x_pos + margin)
            '''
            super(TripletLoss, self).__init__(MarginRankingLoss(margin=margin), device, )
        else:
            '''
            SoftMarginLoss(x, y) = sum( log(1+exp(-y_i*x_i)) )
            '''
            super(TripletLoss, self).__init__(SoftMarginLoss(), device, )

    def _hem_forward(self, code_emb: torch.Tensor, comment_emb: torch.Tensor) -> torch.Tensor:
        '''
        hard example mining
        triplet loss
        ref: https://github.com/LARC-CMU-SMU/ACME/blob/6eb8b1e94f8f3c398d94ca93a69c0d6aafa1d428/triplet_loss.py#L6-L26
        hard example mining
        ref: https://github.com/LARC-CMU-SMU/ACME/blob/6eb8b1e94f8f3c398d94ca93a69c0d6aafa1d428/triplet_loss.py#L52-L89
        '''
        dist_mat = euclidean_dist(comment_emb, code_emb)

        pos_ind = torch.eye(code_emb.size(0)).to(dist_mat.device)
        pos_dist = dist_mat[pos_ind.bool()].unsqueeze(dim=-1)
        neg_dist, _ = dist_mat[(1 - pos_ind).bool()].contiguous().view(code_emb.size(0), -1).min(dim=-1, keepdim=True)

        # y = Variable(input_neg.data.new().resize_as_(input_neg.data).fill_(1)).to(input_pos.device)
        y = Variable(torch.ones_like(neg_dist).to(neg_dist.device))
        if self._margin is not None:
            loss = self._base(neg_dist, pos_dist, y)
        else:
            loss = self._base(neg_dist - pos_dist, y)
        return loss

    def _general_forward(self, code_emb: torch.Tensor, comment_emb: torch.Tensor, ) -> torch.Tensor:
        '''
        average loss
        ref: https://github.com/github/CodeSearchNet/blob/e70e71273e0982af4ea13231dbb26866f5a7c4a1/src/models/model.py#L329-L345
        '''
        dist_mat = euclidean_dist(comment_emb, code_emb)
        # get diagonal elements
        # namely, dist<anchor, pos>
        pos_ind = torch.eye(code_emb.size(0)).to(dist_mat.device)
        pos_dist = dist_mat[pos_ind.bool()].unsqueeze(dim=-1)

        # max(0, dist<anchor, pos> - dist<anchor, neg> + margin)
        pointwise_loss = torch.relu(pos_dist - dist_mat + self._margin)
        # get average triplet loss of <anchor, pos, neg1, neg2...>
        pointwise_loss *= 1 - pos_ind
        # clamp(1) is to avoid pointwise_loss[i,:]=all zero
        loss = pointwise_loss.sum(dim=-1) / (pointwise_loss > 0).sum(dim=-1).clamp(1).float()
        loss = loss.mean()
        return loss

    def forward(self, code_emb: torch.Tensor, comment_emb: torch.Tensor) -> torch.Tensor:
        if self._mining:
            loss = self._hem_forward(code_emb, comment_emb)
        else:
            loss = self._general_forward(code_emb, comment_emb)
        return loss


if __name__ == '__main__':
    loss = TripletLoss(device=True, margin=0.1, mining=True)
    code_emb = torch.randn(3, 2, requires_grad=True).cuda()
    comment_emb = torch.randn(3, 2, requires_grad=True).cuda() + 0.1
    loss_value = loss(code_emb, comment_emb)
    print(loss_value)

    loss = TripletLoss(device=True, margin=0.1, )
    loss_value = loss(code_emb, comment_emb)
    print(loss_value)
