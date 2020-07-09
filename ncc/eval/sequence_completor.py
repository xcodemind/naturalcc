# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch

from ncc.eval import search
from ncc.utils import utils
from ncc.data.tools import data_utils
from ncc.modules.summarization.fairseq_incremental_decoder import FairseqIncrementalDecoder


class SequenceCompletor(object):
    def __init__(
        self,
        tgt_dict,
        # beam_size=1,
        # max_len_a=0,
        # max_len_b=200,
        # min_len=1,
        # normalize_scores=True,
        # len_penalty=1.,
        # unk_penalty=0.,
        retain_dropout=False,
        # temperature=1.,
        # match_source_len=False,
        # no_repeat_ngram_size=0,
        # search_strategy=None,
        eos=None
    ):
        """Generates translations of a given source sentence.

        Args:
            tgt_dict (~fairseq.data.Dictionary): target dictionary
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            retain_dropout (bool, optional): use dropout when generating
                (default: False)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos() if eos is None else eos
        # self.vocab_size = len(tgt_dict)
        # self.beam_size = beam_size
        # # the max beam size is the dictionary size - 1, since we never select pad
        # self.beam_size = min(beam_size, self.vocab_size - 1)
        # self.max_len_a = max_len_a
        # self.max_len_b = max_len_b
        # self.min_len = min_len
        # self.normalize_scores = normalize_scores
        # self.len_penalty = len_penalty
        # self.unk_penalty = unk_penalty
        self.retain_dropout = retain_dropout
        # self.temperature = temperature
        # self.match_source_len = match_source_len
        # self.no_repeat_ngram_size = no_repeat_ngram_size
        # assert temperature > 0, '--temperature must be greater than 0'
        #
        # self.search = (
        #     search.BeamSearch(tgt_dict) if search_strategy is None else search_strategy
        # )


    @torch.no_grad()
    def complete(self, models, sample, **kwargs):
        """Generate a batch of translations.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        model = EnsembleModel(models)
        return self._complete(model, sample, **kwargs)

    @torch.no_grad()
    def _complete(
        self,
        model,
        sample,
        # prefix_tokens=None,
        # bos_token=None,
        **kwargs
    ):
        if not self.retain_dropout:
            model.eval()

        net_output = model(**sample['net_input'])
        return net_output

class EnsembleModel(torch.nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.incremental_states = None
        if all(hasattr(m, 'decoder') and isinstance(m.decoder, FairseqIncrementalDecoder) for m in models):
            self.incremental_states = {m: {} for m in models}

    @torch.no_grad()
    def forward(self, src_tokens, **kwargs):
        """
        Run the forward pass for a decoder-only model.

        Feeds a batch of tokens through the decoder to predict the next tokens.

        Args:
            src_tokens (LongTensor): tokens on which to condition the decoder,
                of shape `(batch, tgt_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, seq_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        if len(self.models) == 1:
            return self.models[0](src_tokens, **kwargs)
        for model in zip(self.models):
           pass
