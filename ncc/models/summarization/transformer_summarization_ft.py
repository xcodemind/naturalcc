import torch
import torch.nn as nn
from ncc.models.fairseq_model import FairseqEncoderDecoderModel
from ncc.modules.embedding import Embedding
from ncc.modules.roberta.transformer_sentence_encoder import TransformerSentenceEncoder
from ncc.modules.seq2seq.transformer_decoder import TransformerDecoder
from ncc.models import register_model
from ncc.models.codebert.code_roberta import RobertaEncoder
from ncc.modules.code2vec.fairseq_encoder import EncoderOut
from ncc.utils import utils
DEFAULT_MAX_SOURCE_POSITIONS = 1e5
DEFAULT_MAX_TARGET_POSITIONS = 1e5


@register_model('transformer_summarization_ft')
class TransformerFtModel(FairseqEncoderDecoderModel):
    """
        Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
        <https://arxiv.org/abs/1706.03762>`_.

        Args:
            encoder (TransformerEncoder): the encoder
            decoder (TransformerDecoder): the decoder

        The Transformer model provides the following named architectures and
        command-line arguments:

        .. argparse::
            :ref: fairseq.models.transformer_parser
            :prog:
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.supports_align_args = True

    @classmethod
    def build_model(cls, args, config, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        # base_architecture(args)
        if 'max_positions' not in args['model']:
            args['model']['max_positions'] = args['task']['tokens_per_sample']

        if args['model']['encoder_layers_to_keep']:
            args['model']['encoder_layers'] = len(args['model']['encoder_layers_to_keep'].split(","))
        if args['model']['decoder_layers_to_keep']:
            args['model']['decoder_layers'] = len(args['model']['decoder_layers_to_keep'].split(","))

        if args['model']['max_source_positions'] is None:
            args['model']['max_source_positions'] = DEFAULT_MAX_SOURCE_POSITIONS
        if args['model']['max_target_positions'] is None:
            args['model']['max_target_positions'] = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        # if args['model']['share_all_embeddings']:
        #     if src_dict != tgt_dict:
        #         raise ValueError("--share-all-embeddings requires a joined dictionary")
        #     if args['model']['encoder_embed_dim'] != args['model']['decoder_embed_dim']:
        #         raise ValueError(
        #             "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
        #         )
        #     if args['model']['decoder_embed_path'] and (
        #             args['model']['decoder_embed_path'] != args['model']['encoder_embed_path']
        #     ):
        #         raise ValueError(
        #             "--share-all-embeddings not compatible with --decoder-embed-path"
        #         )
        #     encoder_embed_tokens = build_embedding(
        #         src_dict, args['model']['encoder_embed_dim'], args['model']['encoder_embed_path']
        #     )
        #     decoder_embed_tokens = encoder_embed_tokens
        #     args['model']['share_decoder_input_output_embed'] = True
        # else:
        #     encoder_embed_tokens = build_embedding(
        #         src_dict, args['model']['encoder_embed_dim'], args['model']['encoder_embed_path']
        #     )
        #     decoder_embed_tokens = build_embedding(
        #         tgt_dict, args['model']['decoder_embed_dim'], args['model']['decoder_embed_path']
        #     )

        # encoder = cls.build_encoder(args, src_dict)
        encoder = RobertaEncoder(args, src_dict)
        if args['model']['share_all_embeddings']:
            decoder_embed_tokens = encoder.sentence_encoder.embed_tokens
        else:
            decoder_embed_tokens = build_embedding(
                tgt_dict, args['model']['decoder_embed_dim'], args['model']['decoder_embed_path']
            )
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)

        return cls(args, encoder, decoder)

    # @classmethod
    # def build_encoder(cls, args, src_dict):
    #     # return TransformerEncoder(args, src_dict, embed_tokens)
    #     sentence_encoder = TransformerSentenceEncoder(
    #         padding_idx=src_dict.pad(),
    #         vocab_size=len(src_dict),
    #         num_encoder_layers=args['model']['encoder_layers'],
    #         embedding_dim=args['model']['encoder_embed_dim'],
    #         ffn_embedding_dim=args['model']['encoder_ffn_embed_dim'],
    #         num_attention_heads=args['model']['encoder_attention_heads'],
    #         dropout=args['model']['dropout'],
    #         attention_dropout=args['model']['attention_dropout'],
    #         activation_dropout=args['model']['activation_dropout'],
    #         layerdrop=args['model']['encoder_layerdrop'],
    #         max_seq_len=args['model']['max_positions'],
    #         num_segments=0,
    #         encoder_normalize_before=True,
    #         apply_bert_init=True,
    #         activation_fn=args['model']['activation_fn'],
    #     )
    #     return sentence_encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra, padding_mask = self.encoder(src_tokens, features_only=True, src_lengths=src_lengths, **kwargs)
        from torch import Tensor
        from typing import NamedTuple
        # encoder_out = EncoderOut(
        #     encoder_out=x,  # T x B x C
        #     # encoder_padding_mask=encoder_padding_mask,  # B x T
        #     # encoder_embedding=encoder_embedding,  # B x T x C
        #     # encoder_states=encoder_states,  # List[T x B x C]
        # )
        EncoderOut2 = NamedTuple(
            "EncoderOut",
            [
                ("encoder_out", Tensor),  # T x B x C
                ("encoder_padding_mask", Tensor),  # B x T
            ],
        )
        encoder_out = EncoderOut2(encoder_out=x, encoder_padding_mask=padding_mask)
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return decoder_out