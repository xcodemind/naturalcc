import torch
import torch.nn as nn
from ncc.modules.completion.layer_norm import LayerNorm
from ncc.modules.attention.path_multihead_attention import PathMultiheadAttention
import math


def gelu(x):
    return (
        0.5
        * x
        * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    )

class MLP(nn.Module):
    def __init__(self, n_state, n_embd):
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(n_embd, n_state)
        self.c_proj = nn.Linear(n_state, n_embd)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        n_ctx,
        n_head,
        n_embd,
        layer_norm_epsilon,
        scale=False,
        rel_vocab_size=None,
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.ln_1 = LayerNorm(n_embd, std_eps=layer_norm_epsilon)
        self.attn = PathMultiheadAttention(
            n_embd, n_ctx, n_head, scale, rel_vocab_size
        )
        self.ln_2 = LayerNorm(n_embd, std_eps=layer_norm_epsilon)
        self.mlp = MLP(4 * n_embd, n_embd)

    def forward(self, x, rel):
        a = self.attn(self.ln_1(x), rel)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x