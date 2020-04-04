# -*- coding: utf-8 -*-
import numpy as np
import torch.nn as nn
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from ncc import LOGGER
from ncc.models.template import *
from ncc.metric import BaseLoss
from ncc.models.retrieval.ahn.util import *
from ncc.module.code2vec.base.emb import Encoder_Emb
from typing import Dict, Any, List


class AHN_NBOW(CodeEnc_CmntEnc):
    def __init__(self, args: Dict, TRAIN_NUM: int, gamma=1, eta=1, ):
        super(AHN_NBOW, self).__init__(
            args=args,
            # code_encoder=CodeEnocder_MMAN(args),
            code_encoder=Encoder_Emb.load_from_config(args, modal='code_tokens'),
            comment_encoder=Encoder_Emb.load_from_config(args, modal='comment'),
        )
        self.TRAIN_NUM = TRAIN_NUM
        self.BATCH_SIZE = self.args['training']['batch_size']
        self.BIT = self.args['hash']['bit']
        self.gamma = gamma
        self.eta = eta

        # discriminator for code modal
        # if modal from code, return True; else(modal from comment), return false
        self.code_discriminator = nn.Sequential(
            nn.Linear(self.args['training']['embed_size'], 512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(256, 1),
        )

        # hash
        self.hash_encoder = nn.Sequential(
            nn.Linear(self.args['training']['embed_size'], self.BIT),
            # nn.Tanh()
        )

        # self.label = torch.eye(self.TRAIN_NUM)
        # CODE/COMMENT buffer for temporarily store code/comment hash code
        self.code_buffer = torch.randn(self.TRAIN_NUM, self.BIT).float()
        self.cmnt_buffer = torch.randn(self.TRAIN_NUM, self.BIT).float()
        # binary code
        self.binary_hash = torch.sign(self.code_buffer + self.cmnt_buffer).float()
        # const variables
        self.ONES = torch.ones(self.BATCH_SIZE, 1)
        self.ONES_ = torch.ones(self.TRAIN_NUM - self.BATCH_SIZE, 1)

        if args['common']['device'] is not None:
            # self.label = self.label.cuda()
            self.code_buffer = self.code_buffer.cuda()
            self.cmnt_buffer = self.cmnt_buffer.cuda()
            self.binary_hash = self.binary_hash.cuda()
            self.ONES = self.ONES.cuda()
            self.ONES_ = self.ONES_.cuda()

    def train_code_layers(self, batch_data: Dict, code_optimizer: torch.optim, ):
        update_ind, unupdate_ind = batch_data['index']
        # LOGGER.info(update_ind.size())
        # update_label = self.label[update_ind]
        # S = calc_sim_mat(update_label, self.label)
        S = one_hot_encode(update_ind, self.TRAIN_NUM)

        # code -> encoder -> hash_encoder -> code_hash
        code_emb = self.code_forward(batch_data)
        code_hash = self.hash_encoder(code_emb)
        # LOGGER.info('code_hash[0]: {}'.format(code_hash[0]))

        self.code_buffer[update_ind, :] = code_hash.data.detach()
        LOGGER.info(self.code_buffer[update_ind, :][0])

        theta = 0.5 * torch.matmul(code_hash, self.cmnt_buffer.t())
        # LOGGER.info(S.size())
        # LOGGER.info(theta.size())
        neglog_loss = -(S * theta - torch.log(1.0 + torch.exp(theta))).sum()
        # LOGGER.info('neglog_loss: {}'.format(neglog_loss))
        quantization_loss = ((self.binary_hash[update_ind, :] - code_hash) ** 2).sum()
        # LOGGER.info('quantization_loss: {}'.format(quantization_loss))
        balance_loss = (
                (torch.mm(code_hash.t(), self.ONES) + torch.mm(self.code_buffer[unupdate_ind].t(), self.ONES_)) ** 2
        ).sum()
        # LOGGER.info('balance_loss: {}'.format(balance_loss))
        loss = neglog_loss + self.gamma * quantization_loss + self.eta * balance_loss
        # loss = neglog_loss
        # LOGGER.info('loss: {}'.format(loss.size()))
        loss /= (self.BATCH_SIZE * self.TRAIN_NUM)
        LOGGER.info('loss: {:.4f}'.format(loss.item()))

        code_optimizer.zero_grad()
        loss.backward()
        if self.args['sl']['max_grad_norm'] != -1:
            nn.utils.clip_grad_norm_(self.code_parameters(), self.args['sl']['max_grad_norm'])
        code_optimizer.step()
        # LOGGER.info(self.code_buffer[update_ind, :][0])
        return loss

    def train_comment_layers(self, batch_data: Dict, cmnt_optimizer: torch.optim, ):
        update_ind, unupdate_ind = batch_data['index']
        # LOGGER.info(update_ind.size())
        # update_label = self.label[update_ind]
        # S = calc_sim_mat(update_label, self.label)
        S = one_hot_encode(update_ind, self.TRAIN_NUM)

        # comment -> encoder -> hash_encoder -> comment_hash
        comment_emb = self.comment_forward(batch_data)
        comment_hash = self.hash_encoder(comment_emb)
        self.cmnt_buffer[update_ind, :] = comment_hash.data

        theta = 0.5 * torch.matmul(comment_hash, self.code_buffer.t())
        try:
            neglog_loss = -(S * theta - torch.log(1.0 + torch.exp(theta))).sum()
        except:
            print(S.size())
            print(theta.size())
            print(S)
            print(theta)
            S * theta
            torch.log(1.0 + torch.exp(theta))
            exit()
        quantization_loss = torch.sum(torch.pow(self.binary_hash[update_ind, :] - comment_hash, 2))
        balance_loss = torch.sum(
            torch.pow(comment_hash.t().mm(self.ONES) + self.cmnt_buffer[unupdate_ind].t().mm(self.ONES_), 2))
        loss = neglog_loss + self.gamma * quantization_loss + self.eta * balance_loss
        loss /= (self.BATCH_SIZE * self.TRAIN_NUM)
        # LOGGER.info('loss: {}'.format(loss))

        cmnt_optimizer.zero_grad()
        loss.backward()
        if self.args['sl']['max_grad_norm'] != -1:
            nn.utils.clip_grad_norm_(self.cmnt_parameters(), self.args['sl']['max_grad_norm'])
        cmnt_optimizer.step()
        return loss

    def update_hash_mat(self):
        self.binary_hash = torch.sign(self.code_buffer + self.cmnt_buffer)

    def calc_total_loss(self):
        Sim = torch.eye(self.TRAIN_NUM)
        theta = 0.5 * torch.matmul(self.code_buffer.cpu(), self.cmnt_buffer.t().cpu())
        neglog_loss = torch.sum(torch.log(1 + torch.exp(theta)) - Sim * theta).cuda()
        quantization_loss = torch.sum(
            torch.pow(self.binary_hash - self.code_buffer, 2) + torch.pow(self.binary_hash - self.cmnt_buffer, 2)
        )
        balance_loss = torch.sum(torch.pow(self.code_buffer.sum(dim=0), 2) + torch.pow(self.cmnt_buffer.sum(dim=0), 2))
        loss = neglog_loss + self.gamma * quantization_loss + self.eta * balance_loss
        loss /= (self.TRAIN_NUM * self.BATCH_SIZE)
        return loss

    def code_forward(self, batch_data: Dict, ) -> torch.Tensor:
        code = batch_data['code_tokens'][0]
        code_emb = self.code_encoder(code)
        code_emb = torch.tanh(code_emb)
        code_emb, _ = code_emb.max(dim=1)
        code_emb = norm(code_emb)
        # code_emb = torch.relu(code_emb)
        return code_emb

    def comment_forward(self, batch_data: Dict, ) -> torch.Tensor:
        comment = batch_data['docstring_tokens'][0]
        cmnt_emb = self.comment_encoder(comment)
        cmnt_emb = torch.tanh(cmnt_emb)
        cmnt_emb, _ = cmnt_emb.max(dim=1)
        cmnt_emb = norm(cmnt_emb)
        # cmnt_emb = torch.relu(cmnt_emb)
        return cmnt_emb

    def hamming_loss(self, batch_data: Dict, print_hash=False, ):
        code_emb = self.code_forward(batch_data)
        code_hash = self.hash_encoder(code_emb)
        code_hash = torch.sign(code_hash)
        comment_emb = self.comment_forward(batch_data)
        comment_hash = self.hash_encoder(comment_emb)
        comment_hash = torch.sign(comment_hash)
        if print_hash:
            LOGGER.info(code_hash[0].tolist())
            LOGGER.info(comment_hash[0].tolist())
        dist = hamming_dist(code_hash, comment_hash)
        return dist

    def train_sl(self, batch_data: Dict, criterion: BaseLoss, ) -> torch.Tensor:
        code_emb = self.code_forward(batch_data)
        comment_emb = self.comment_forward(batch_data)
        loss = criterion(code_emb, comment_emb)
        return loss

    def _train_discriminator(self, code_emb: torch.Tensor, comment_emb: torch.Tensor, optimizer: Optimizer, ) -> None:
        def _compute_gradient_penalty(discriminator: Module, real_samples: torch.Tensor,
                                      fake_samples: torch.Tensor, ) -> torch.Tensor:
            """
            Calculates the gradient penalty loss for WGAN-GP
            ref: https://github.com/LARC-CMU-SMU/ACME/blob/6eb8b1e94f8f3c398d94ca93a69c0d6aafa1d428/train.py#L196-L215
            """
            # Random weight term for interpolation between real and fake samples
            alpha = torch.Tensor(np.random.random((real_samples.size(0), 1))).float().to(real_samples.device)
            # Get random interpolation between real and fake samples
            interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
            d_interpolates = discriminator(interpolates)
            fake = torch.autograd.Variable(
                torch.Tensor(real_samples.shape[0], 1).fill_(1.0).float().to(real_samples.device),
                requires_grad=False)
            # Get gradient w.r.t. interpolates
            gradients = torch.autograd.grad(
                outputs=d_interpolates,  # fack samples
                inputs=interpolates,  # real samples
                grad_outputs=fake,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            gradients = gradients.view(gradients.size(0), -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            return gradient_penalty

        # detach from graph -> to train discriminator only
        real_validity = self.code_discriminator(code_emb.detach())
        fake_validity = self.code_discriminator(comment_emb.detach())
        # LOGGER.info('real loss: {:.4f}, fake loss: {:.4f}'.format(-torch.mean(real_validity), torch.mean(fake_validity)))
        gradient_penalty = _compute_gradient_penalty(self.code_discriminator, code_emb.detach(), comment_emb.detach())
        al_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gradient_penalty
        # LOGGER.info('DM loss: {:.4f}'.format(al_loss))
        optimizer.zero_grad()
        al_loss.backward()
        optimizer.step()

    def train_al(self, batch_data: Dict, criterion: BaseLoss, disc_optimizer: Optimizer, ):
        code_emb = self.code_forward(batch_data)
        comment_emb = self.comment_forward(batch_data)
        self._train_discriminator(code_emb, comment_emb, disc_optimizer)

        # generated code loss, will be used later
        # fake_code_validity = self.code_discriminator(comment_emb)
        # fake_loss = -torch.mean(fake_code_validity)
        # LOGGER.info('DM loss: {:.4f}'.format(fake_loss.item()))

        # code_emb = self.hash_encoder(code_emb)
        # comment_emb = self.hash_encoder(comment_emb)
        loss = criterion(code_emb, comment_emb)
        return loss

    def disc_parameters(self) -> Any:
        return self.code_discriminator.parameters()

    def trainable_parameters(self) -> List:
        return list(self.code_encoder.parameters()) + \
               list(self.comment_encoder.parameters())

    def code_parameters(self) -> List:
        return list(self.code_encoder.parameters()) + list(self.hash_encoder.parameters())

    def cmnt_parameters(self) -> List:
        return list(self.comment_encoder.parameters()) + list(self.hash_encoder.parameters())
