import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, wemb, opt): #code_encoder
        super(Critic, self).__init__()
        self.opt = opt
        self.rnn = getattr(nn, opt.rnn_type)(opt.ninp, opt.nhid, opt.nlayers, dropout=opt.dropout)
        self.linear = nn.Linear(sum(num_filters), 2)
        self.wemb = wemb

    def forward(self, inputs, hidden):
        emb_enc = self.wemb(inputs.clone()[:, :-1])
        _, hidden = self.rnn(emb_enc, hidden)
        out = self.linear(F.dropout(hidden[0][-1], self.opt.dropout, training=self.training))

        return out