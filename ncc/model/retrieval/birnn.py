# -*- coding: utf-8 -*-

import sys

sys.path.append('./')

from ncc import *
from ncc.model.template import *
from ncc.module.code2vec.encoder_tok import Encoder_EmbRNN
from ncc.metric import *


class BiRNN(CodeEnc_CmntEnc):

    def __init__(self, config: Dict):
        super(BiRNN, self).__init__(
            config=config,
            code_encoder=Encoder_EmbRNN.load_from_config(config, modal='tok'),
            comment_encoder=Encoder_EmbRNN.load_from_config(config, modal='tok'),
        )

    def code_forward(self, batch_data: Dict, ) -> Any:
        code, code_len, code_mask = batch_data['tok']
        hidden = self.code_encoder.init_hidden(code.size(0))
        output, (h, _) = self.code_encoder(code, input_len=code_len, hidden=hidden)

        output = torch.tanh(output)
        output, _ = output.max(dim=1)
        return output

        # code_emb = h.transpose(0, 1).reshape(h.size(1), -1)
        # return code_emb

    def comment_forward(self, batch_data: Dict, ) -> Any:
        comment, _, _, comment_len, _ = batch_data['comment']
        hidden = self.comment_encoder.init_hidden(comment.size(0))
        output, (h, _) = self.comment_encoder(comment, input_len=comment_len, hidden=hidden)

        output = torch.tanh(output)
        output, _ = output.max(dim=1)
        return output

        # comment_emb = h.transpose(0, 1).reshape(h.size(1), -1)
        # return comment_emb

    def train_sl(self, batch_data: Dict, criterion: BaseLoss, ) -> torch.Tensor:
        code_emb = self.code_forward(batch_data)
        comment_emb = self.comment_forward(batch_data)
        loss = criterion(code_emb, comment_emb)
        return loss


if __name__ == '__main__':
    pass

    # Evaluator for retrieval
