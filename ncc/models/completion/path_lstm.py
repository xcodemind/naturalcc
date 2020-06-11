import torch
import torch.nn as nn


# base RNN model
class LSTMModel(torch.nn.Module):
    def __init__(self, vocab_size, n_embd, loss_fn, n_ctx):
        super(LSTMModel, self).__init__()
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