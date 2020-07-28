import torch
import torch.nn as nn
from ncc.models.fairseq_model import FairseqEncoderDecoderModel
from ncc.modules.embedding import Embedding
from ncc.modules.code2vec.transformer_encoder import TransformerEncoder
from ncc.modules.seq2seq.transformer_decoder import TransformerDecoder
from ncc.models import register_model
from ncc.utils import utils
DEFAULT_MAX_SOURCE_POSITIONS = 1e5
DEFAULT_MAX_TARGET_POSITIONS = 1e5


@register_model('transformer_summarization')
class TransformerModel(FairseqEncoderDecoderModel):
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
    #
    # @staticmethod
    # def add_args(parser):
    #     """Add model-specific arguments to the parser."""
    #     # fmt: off
    #     parser.add_argument('--dropout', type=float, metavar='D',
    #                         help='dropout probability')
    #     parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
    #                         help='encoder embedding dimension')
    #     parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
    #                         help='path to pre-trained encoder embedding')
    #     parser.add_argument('--encoder-freeze-embed', action='store_true',
    #                         help='freeze encoder embeddings')
    #     parser.add_argument('--encoder-hidden-size', type=int, metavar='N',
    #                         help='encoder hidden size')
    #     parser.add_argument('--encoder-layers', type=int, metavar='N',
    #                         help='number of encoder layers')
    #     parser.add_argument('--encoder-bidirectional', action='store_true',
    #                         help='make all layers of encoder bidirectional')
    #     parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
    #                         help='decoder embedding dimension')
    #     parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
    #                         help='path to pre-trained decoder embedding')
    #     parser.add_argument('--decoder-freeze-embed', action='store_true',
    #                         help='freeze decoder embeddings')
    #     parser.add_argument('--decoder-hidden-size', type=int, metavar='N',
    #                         help='decoder hidden size')
    #     parser.add_argument('--decoder-layers', type=int, metavar='N',
    #                         help='number of decoder layers')
    #     parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
    #                         help='decoder output embedding dimension')
    #     parser.add_argument('--decoder-attention', type=str, metavar='BOOL',
    #                         help='decoder attention')
    #     parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
    #                         help='comma separated list of adaptive softmax cutoff points. '
    #                              'Must be used with adaptive_loss criterion')
    #     parser.add_argument('--share-decoder-input-output-embed', default=False,
    #                         action='store_true',
    #                         help='share decoder input and output embeddings')
    #     parser.add_argument('--share-all-embeddings', default=False, action='store_true',
    #                         help='share encoder, decoder and output embeddings'
    #                              ' (requires shared dictionary and embed dim)')
    #
    #     # Granular dropout settings (if not specified these default to --dropout)
    #     parser.add_argument('--encoder-dropout-in', type=float, metavar='D',
    #                         help='dropout probability for encoder input embedding')
    #     parser.add_argument('--encoder-dropout-out', type=float, metavar='D',
    #                         help='dropout probability for encoder output')
    #     parser.add_argument('--decoder-dropout-in', type=float, metavar='D',
    #                         help='dropout probability for decoder input embedding')
    #     parser.add_argument('--decoder-dropout-out', type=float, metavar='D',
    #                         help='dropout probability for decoder output')
    #     # fmt: on

    @classmethod
    def build_model(cls, args, config, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        # base_architecture(args)

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

        if args['model']['share_all_embeddings']:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args['model']['encoder_embed_dim'] != args['model']['decoder_embed_dim']:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args['model']['decoder_embed_path'] and (
                    args['model']['decoder_embed_path'] != args['model']['encoder_embed_path']
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = build_embedding(
                src_dict, args['model']['encoder_embed_dim'], args['model']['encoder_embed_path']
            )
            decoder_embed_tokens = encoder_embed_tokens
            args['model']['share_decoder_input_output_embed'] = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args['model']['encoder_embed_dim'], args['model']['encoder_embed_path']
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args['model']['decoder_embed_dim'], args['model']['decoder_embed_path']
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )