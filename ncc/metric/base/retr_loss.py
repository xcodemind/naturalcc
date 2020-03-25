# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from ncc import *
from ncc.metric import *

'''
this file contains losses from Github: CodeSearchNet
1) negative log loss
2) triple loss
'''


class RetrievalNLLoss(BaseLoss):
    '''
    information entropy for retrieval
    ref: https://github.com/github/CodeSearchNet/blob/e70e71273e0982af4ea13231dbb26866f5a7c4a1/src/models/model.py#L284-L297
    '''

    def __init__(self, device: bool, ):
        super(RetrievalNLLoss, self).__init__(
            base=nn.NLLLoss(),
            device=device,
        )

    def forward(self, code_emb: torch.Tensor, comment_emb: torch.Tensor, ) -> torch.Tensor:
        logits = torch.mm(comment_emb, code_emb.t())

        loss = self._base(
            F.log_softmax(logits, dim=-1),
            torch.arange(code_emb.size(0)).to(code_emb.device),
        )
        return loss




if __name__ == '__main__':
    batch_size, len = 4, 10
    code_emb = torch.randn(batch_size, len, requires_grad=True).cuda()
    comment_emb = torch.randn(batch_size, len, requires_grad=True).cuda()

    entropy_loss = RetrievalNLLoss(device=True)
    loss = entropy_loss(code_emb, comment_emb)
    print(loss.item())
