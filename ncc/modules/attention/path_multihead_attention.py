import math
import torch
import torch.nn as nn


class PathMultiheadAttention(nn.Module):
    def __init__(
        self, nx, n_ctx, n_head, scale=False, rel_vocab_size=None
    ):
        super(PathMultiheadAttention, self).__init__()
        n_state = nx
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % n_head == 0
        self.register_buffer(
            "bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx)
        )
        self.n_head = n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = nn.Linear(nx, n_state * 3)
        self.c_proj = nn.Linear(nx, n_state)

        # if rel exists
        if rel_vocab_size is not None:
            self.rel_weights = nn.Embedding(rel_vocab_size, n_head)

    def _attn(self, q, k, v, rel=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        # b = self.bias[:, :, ns - nd : ns, :ns] # TODO
        # w = w * b - 1e10 * (1 - b)

        # # add in more tree structure
        # if rel is not None:
        #     w = w * (rel * b)

        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, rel=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if rel is not None:
            rel = self.rel_weights(rel)
            rel = rel.permute(0, 3, 1, 2)

        # self attention component
        a = self._attn(query, key, value, rel)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a