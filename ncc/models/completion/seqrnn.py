# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import torch.nn as nn
from ncc.models import register_model
from ncc.models.fairseq_model import FairseqLanguageModel
from ncc.modules.summarization.fairseq_decoder import FairseqDecoder
from ncc.modules.summarization.lstm_decoder import LSTMDecoder

logger = logging.getLogger(__name__)


@register_model('seqrnn')
class SeqRNNModel(FairseqLanguageModel):
    def __init__(self, args, decoder):
        super().__init__(decoder)
        self.args = args

    @classmethod
    def build_model(cls, args, config, task):
        # decoder = LSTMEncoder(args, task.source_dictionary)
        decoder = LSTMDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args['model']['decoder_embed_dim'],
            hidden_size=args['model']['decoder_hidden_size'],
            out_embed_dim=args['model']['decoder_out_embed_dim'],
            num_layers=args['model']['decoder_layers'],
            dropout_in=args['model']['decoder_dropout_in'],
            dropout_out=args['model']['decoder_dropout_out'],
            attention=args['model']['decoder_attention'],
            encoder_output_units=0,
            # pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=args['model']['share_decoder_input_output_embed'],
            adaptive_softmax_cutoff=(
                args['model']['adaptive_softmax_cutoff']
                if args['criterion'] == 'adaptive_loss' else None
            ),
            # max_target_positions=max_target_positions
        )
        return cls(args, decoder)


class SeqRNNModel_(FairseqLanguageModel):
    def __init__(self, vocab_size, n_embd, loss_fn, n_ctx):
        super(SeqRNNModel_, self).__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.lstm = nn.LSTM(n_embd, n_embd, num_layers=1, dropout=0.5, batch_first=True)
        self.decoder = nn.Linear(n_embd, vocab_size)

        self.loss_fn = loss_fn
        self.half_ctx = int(n_ctx / 2)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, x, y, ext=None, rel=None, paths=None, return_loss=False
    ):
        embed = self.embedding(x)  # bs, max_len, n_embd
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(embed)  # bs, max_len, n_embd
        y_pred = self.decoder(lstm_out)  # bs, max_len, vocab_size
        if not return_loss:
            return y_pred

        # ext contains a list of idx of where to take the loss from
        # we linearize it first
        ids = []
        max_len = y.size(-1)
        for i, ext_i in enumerate(ext):
            ids += [i * max_len + j for j in range(ext_i, max_len)]
        loss = self.loss_fn(y_pred.view(-1, y_pred.size(-1))[ids], y.view(-1)[ids])
        return loss