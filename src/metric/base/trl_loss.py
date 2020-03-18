# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

from src import *
from src.metric import *
from src.utils.constants import PAD


class _TRLLoss(Module):
    '''
    reference: Deep transfer reinforcement learning for text summarization
    CE loss= cross_entropy() # target data, eg. c#
    TRL loss = -1 * sum(
                        (1 - eta) * log(greedy_probability) * (greedy_reward - sampled_reward) # target data, eg.c#
                        - eta * log(greedy_probability) * (greedy_reward - sampled_reward) # source data, eg.c
                        )
    '''

    def __init__(self, eta=0.1, zeta=0.1):
        super(_TRLLoss, self).__init__()
        self.eta = eta
        self.zeta = zeta

    def CE_Loss(self, log_probs: torch.Tensor, target: torch.Tensor, ) -> torch.Tensor:
        '''
        :param lprobs: [batch_size, seq_len, probability_size]
        :param target: [batch_size, seq_len]
        copy from LMCriterion
        '''
        log_probs = log_probs.reshape(-1, log_probs.size(-1))
        target = target.reshape(-1, 1)
        log_probs_select = torch.gather(log_probs, 1, target)
        mask = target.data.gt(0)  # generate the mask
        out = torch.masked_select(log_probs_select, mask)
        ce_loss = -torch.sum(out) / torch.sum(target.data.ne(PAD)).float()  # get the average loss.
        return ce_loss

    def RL_Loss(self, log_probs: torch.Tensor, target: torch.Tensor, reward_diff: torch.Tensor, ) -> torch.Tensor:
        log_probs_select = torch.gather(log_probs, -1, target.unsqueeze(-1)).squeeze(-1)
        mask = target.data.gt(0)  # generate the mask
        out = mask.float() * -log_probs_select * reward_diff
        rl_loss = (out.sum(-1) / target.data.ne(PAD).sum(-1).float()).mean()
        return rl_loss

    def forward(self, log_probs: torch.Tensor, target: torch.Tensor, reward_diff: torch.Tensor, ) -> torch.Tensor:
        if self.eta == 0.0:
            ce_loss = self.CE_Loss(log_probs, target)  # crossy entropy for all
            return ce_loss
        else:
            batch_size = log_probs.size(0) // 2
            # log probability
            src_log_probs, trg_log_probs = log_probs[:batch_size, ...], log_probs[batch_size:, ...]
            # target
            src_target, trg_target = target[:batch_size, ...], target[batch_size:, ...]
            # reward difference
            src_reward_diff, trg_reward_diff = reward_diff[:batch_size, ...], reward_diff[batch_size:, ...]

            trg_ce_loss = self.CE_Loss(trg_log_probs, trg_target)
            src_rl_loss = self.RL_Loss(src_log_probs, src_target, src_reward_diff)  # source, eg. c, java, python
            trg_rl_loss = self.RL_Loss(trg_log_probs, trg_target, trg_reward_diff)  # target, eg c#

            return (1.0 - self.eta) * trg_ce_loss + self.eta * (
                    (1.0 - self.zeta) * trg_rl_loss + self.zeta * src_rl_loss
            )


class TRLLoss(BaseLoss):

    def __init__(self, device: bool, eta=0.1, zeta=0.1) -> None:
        eta = 0.1 if eta is None else eta
        zeta = 0.1 if zeta is None else zeta
        super(TRLLoss, self).__init__(_TRLLoss(eta, zeta), device, )


if __name__ == '__main__':
    criterion = TRLLoss(device=True)
    print(criterion)
