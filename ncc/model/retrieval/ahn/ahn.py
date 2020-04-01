# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

from ncc import *
from ncc.model.template import *
from ncc.module.code2vec.encoder_tok import *
from ncc.metric import *
from ncc.module.code2vec.multi_modal.mman_encoder import CodeEnocder_MM


class AHN(CodeEnc_CmntEnc):

    def __init__(self, config: Dict, TRAIN_NUM: int, ):
        super(AHN, self).__init__(
            config=config,
            # code_encoder=CodeEnocder_MMAN(config),
            code_encoder=Encoder_EmbRNN.load_from_config(config, modal='tok'),
            comment_encoder=Encoder_EmbRNN.load_from_config(config, modal='comment'),
        )
        self.TRAIN_NUM = TRAIN_NUM
        self.BATCH_SIZE = self.config['training']['batch_size']
        self.BIT = self.config['hash']['bit']

        # discriminator for code modal
        # if modal from code, return True; else(modal from comment), return false
        self.code_discriminator = nn.Sequential(
            nn.Linear(self.config['training']['rnn_hidden_size'], 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(256, 64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(64, 1),
        )
        # feed a modal feature, return a corresponding hash code
        self.hash_encoder = nn.Sequential(
            nn.Linear(self.config['training']['rnn_hidden_size'], self.config['hash']['bit'], ),
            nn.BatchNorm1d(self.config['hash']['bit']),
            nn.Tanh(),
        )
        # CODE/COMMENT buffer for temporarily store code/comment hash code
        self.CODE_buffer = nn.Parameter(torch.randn(self.TRAIN_NUM, self.BIT), requires_grad=False)
        self.CMNT_buffer = nn.Parameter(torch.randn(self.TRAIN_NUM, self.BIT), requires_grad=False)
        # binary code
        self.binary_hash = nn.Parameter(torch.sign(self.CODE_buffer + self.CMNT_buffer).float(), requires_grad=False)
        # const variables
        self.ONES = nn.Parameter(torch.ones(self.BATCH_SIZE, 1), requires_grad=False)
        self.ONES_ = nn.Parameter(torch.ones(self.TRAIN_NUM - self.BATCH_SIZE, 1), requires_grad=False)

    def _code_forward(self, batch_data: Dict, ) -> torch.Tensor:
        code, code_len, _ = batch_data['tok']
        code_emb, hidden = self.code_encoder(code, code_len)
        code_emb, _ = code_emb.max(dim=1)
        return code_emb

    def code_forward(self, batch_data: Dict, ) -> torch.Tensor:
        code_emb = self._code_forward(batch_data)
        code_emb = self.hash_encoder(code_emb)
        return code_emb

    def _comment_forward(self, batch_data: Dict, ) -> torch.Tensor:
        comment, _, _, comment_len, _ = batch_data['comment']
        comment_emb, (h, _) = self.comment_encoder(comment, comment_len)
        comment_emb, _ = comment_emb.max(dim=1)
        return comment_emb
        # h = h.transpose(dim0=0, dim1=1)
        # h = torch.tanh(h.view(h.size(0), -1))
        # return h

    def comment_forward(self, batch_data: Dict, ) -> torch.Tensor:
        cmnt_emb = self._comment_forward(batch_data)
        cmnt_emb = self.hash_encoder(cmnt_emb)
        return cmnt_emb

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
        gradient_penalty = _compute_gradient_penalty(self.code_discriminator, code_emb.detach(), comment_emb.detach())
        al_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gradient_penalty
        # LOGGER.debug('train discriminator: {}'.format(al_loss.item()))
        optimizer.zero_grad()
        al_loss.backward()
        optimizer.step()

    def train_al(self, batch_data: Dict, criterion: BaseLoss, disc_optimizer: Optimizer, ):
        code_emb = self._code_forward(batch_data)
        comment_emb = self._comment_forward(batch_data)
        self._train_discriminator(code_emb, comment_emb, disc_optimizer)

        code_emb = self.hash_encoder(code_emb)
        comment_emb = self.hash_encoder(comment_emb)
        loss = criterion(code_emb, comment_emb)
        return loss

    def train_code_encoder(self, code_emb: torch.Tensor, update_ind: torch.Tensor, unupdate_ind: torch.Tensor,
                           batch_similarity: torch.Tensor, optimizer: Optimizer, ) -> None:
        code_hash = self.hash_encoder(code_emb)

        self.zero_grad()
        # update CODE_buffer,binary_hash
        self.CODE_buffer[update_ind] = code_hash.data
        # self.binary_hash[update_ind] = code_hash

        # CODE = self.CODE_buffer
        # CMNT = self.CMNT_buffer

        # LOGGER.debug(self.CODE_buffer)
        # LOGGER.debug(self.CMNT_buffer)

        theta = torch.mm(code_hash, self.CMNT_buffer.t()) / 2
        neg_log_loss = -torch.sum(batch_similarity * theta - torch.log(1 + torch.exp(theta)))
        quantization = torch.sum(torch.pow(self.binary_hash[update_ind] - code_hash, 2))
        balance = torch.sum(torch.pow(
            torch.mm(code_hash.t(), self.ONES) + torch.mm(self.CODE_buffer[unupdate_ind].t(), self.ONES_),
            2))
        loss = neg_log_loss + self.config['gamma'] * quantization + self.config['eta'] * balance
        loss /= self.BATCH_SIZE * self.TRAIN_NUM

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # LOGGER.debug(self.CODE_buffer)
        # LOGGER.debug(self.CMNT_buffer)

    def train_comment_encoder(self, cmnt_emb: torch.Tensor, update_ind: torch.Tensor, unupdate_ind: torch.Tensor,
                              batch_similarity: torch.Tensor, optimizer: Optimizer, ) -> None:
        cmnt_hash = self.hash_encoder(cmnt_emb)

        self.zero_grad()
        # update CODE_buffer,binary_hash
        self.CMNT_buffer[update_ind] = cmnt_hash.data
        # self.binary_hash[update_ind] = cmnt_hash

        # CODE = self.CODE_buffer
        # CMNT = self.CMNT_buffer

        # LOGGER.debug(self.CODE_buffer)
        # LOGGER.debug(self.CMNT_buffer)

        theta = torch.mm(cmnt_hash, self.CODE_buffer.t()) / 2
        neg_log_loss = -torch.sum(batch_similarity * theta - torch.log(1 + torch.exp(theta)))
        quantization = torch.sum(torch.pow(self.binary_hash[update_ind] - cmnt_hash, 2))
        balance = torch.sum(torch.pow(
            torch.mm(cmnt_hash.t(), self.ONES) + torch.mm(self.CODE_buffer[unupdate_ind].t(), self.ONES_),
            2))
        loss = neg_log_loss + self.config['gamma'] * quantization + self.config['eta'] * balance
        loss /= self.BATCH_SIZE * self.TRAIN_NUM

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # LOGGER.debug(self.CODE_buffer)
        # LOGGER.debug(self.CMNT_buffer)

    def update_binary_hash(self) -> None:
        self.binary_hash.copy_(torch.sign(self.CODE_buffer + self.CMNT_buffer).float().data)

    def hash_loss(self) -> torch.Tensor:
        theta = torch.mm(self.CODE_buffer, self.CMNT_buffer.t()) / 2
        neg_log_loss = torch.sum(torch.log(1 + torch.exp(theta)) - \
                                 theta * torch.eye(self.TRAIN_NUM).to(theta.device))
        quantization = torch.sum(
            torch.pow(self.binary_hash - self.CODE_buffer, 2) + torch.pow(self.binary_hash - self.CMNT_buffer, 2)
        )
        balance = torch.sum(torch.pow(self.CODE_buffer.sum(dim=0), 2) + torch.pow(self.CMNT_buffer.sum(dim=0), 2))
        loss = neg_log_loss + self.config['gamma'] * quantization + self.config['eta'] * balance
        # LOGGER.debug(loss)
        loss /= self.TRAIN_NUM * self.BATCH_SIZE
        # LOGGER.debug(self.TRAIN_NUM * self.BATCH_SIZE)
        # LOGGER.debug(loss)
        # assert False
        return loss

    def train_hash(self, batch_data: Dict, criterion: BaseLoss,
                   disc_optimizer: Optimizer, ) -> Any:
        '''
        modal alignment
        ref: https://github.com/LARC-CMU-SMU/ACME/blob/6eb8b1e94f8f3c398d94ca93a69c0d6aafa1d428/train.py#L229-L241

        hashing
        ref:
        '''
        # 1) modal alignment
        code_emb = self.code_encoder(batch_data)
        comment_emb = self.comment_forward(batch_data)

        # 2) hashing
        sim_mat = torch.randn(self.TRAIN_NUM, self.TRAIN_NUM, ).to(code_emb.device)
        F_buffer = torch.randn(self.TRAIN_NUM, self.config['hash_code_len']).to(code_emb.device)
        G_buffer = torch.randn(self.TRAIN_NUM, self.config['hash_code_len']).to(code_emb.device)
        B_mat = torch.sign(F_buffer + G_buffer)  # [N, bit]

    def disc_parameters(self) -> List:
        return list(self.code_discriminator.parameters())

    def code_parameters(self) -> List:
        return list(self.code_encoder.parameters()) + list(self.hash_encoder.parameters())

    def cmnt_parameters(self) -> List:
        return list(self.comment_encoder.parameters()) + list(self.hash_encoder.parameters())

    def trainable_parameters(self) -> List:
        return list(self.code_encoder.parameters()) + \
               list(self.comment_encoder.parameters()) + \
               list(self.hash_encoder.parameters())


if __name__ == '__main__':
    # one_hot = one_hot_encode(torch.LongTensor([[1], [3]]).cuda(), class_num=4)
    # print(one_hot)

    all = torch.arange(10)
    delete = torch.Tensor([1, 2])
