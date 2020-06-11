import copy
import torch.nn as nn
from ncc.modules.summarization.fairseq_decoder import FairseqDecoder
from ncc.modules.completion.transformer_encoder_layer import TransformerEncoderLayer
from ncc.modules.completion.layer_norm import LayerNorm


class PathLSTM(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super(PathLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.LSTM = nn.LSTM(n_embd, n_embd, batch_first=True)

    def forward(self, paths):
        embed = self.embedding(paths)  # bs, max_len, max_path_len, n_embd
        batch_size, bag_size, path_len, n_embd = embed.shape
        _, (h_n, _) = self.LSTM(embed.view(batch_size * bag_size, path_len, n_embd))
        return h_n.permute((1, 0, 2)).view((batch_size, bag_size, -1))


class TransformerEncoder(FairseqDecoder):
    def __init__(
        self,
        vocab_size,
        n_layer,
        n_embd,
        n_ctx,
        n_head,
        layer_norm_epsilon,
        root_paths,
        rel_vocab_size,
            dictionary,
    ):
        # super(GPT2Model, self).__init__()
        super().__init__(dictionary)
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_vocab = vocab_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        if root_paths:
            self.path_lstm = PathLSTM(vocab_size, n_embd)
        block = TransformerEncoderLayer(
            n_ctx,
            n_head,
            n_embd,
            layer_norm_epsilon,
            scale=True,
            rel_vocab_size=rel_vocab_size,
        )
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(n_layer)])
        self.ln_f = LayerNorm(n_embd, std_eps=layer_norm_epsilon)

        # self.lm_head = GPT2LMHead(self.wte.weight, n_embd)

    def max_positions(self):
        """Maximum input length supported by the decoder."""
        return None  # an arbitrary large number TODO

    def forward(self, input_ids, rel=None, paths=None):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        inputs_embeds = self.wte(input_ids)
        path_embeds = self.path_lstm(paths) if paths is not None else 0
        hidden_states = inputs_embeds + path_embeds

        for block in self.h:
            hidden_states = block(hidden_states, rel)
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape)

