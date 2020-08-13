# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ncc.utils import utils
from ncc.models import (
    FairseqMoCoModel,
    register_model,
    # register_model_architecture,
)
from ncc.modules.seq2seq.fairseq_decoder import FairseqDecoder
from ncc.modules.roberta.layer_norm import LayerNorm
from ncc.modules.roberta.transformer_sentence_encoder import TransformerSentenceEncoder
from ncc.models.hub_interface import RobertaHubInterface
from ncc.modules.code2vec.contracode_encoder import CodeEncoder, CodeEncoderLSTM
from ncc import LOGGER


@register_model('contracode_moco')
class ContraCodeMoCo(FairseqMoCoModel):

    @classmethod
    def hub_models(cls):
        return {
            'roberta.base': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz',
            'roberta.large': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz',
            'roberta.large.mnli': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.mnli.tar.gz',
            'roberta.large.wsc': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.wsc.tar.gz',
        }

    def __init__(self, args, q_encoder, k_encoder):
        super().__init__(q_encoder, k_encoder)
        self.args = args
        self.K = args['model']['moco_num_keys']
        self.m = args['model']['moco_momentum']
        self.T = args['model']['moco_temperature']
        # # We follow BERT's random weight initialization
        # self.apply(init_bert_params)

        # self.classification_heads = nn.ModuleDict()
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.register_buffer("queue", torch.randn(args['model']['encoder_hidden_size_q'], self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    @classmethod
    def build_model(cls, args, config, task):
        """Build a new model instance."""

        # make sure all arguments are present
        # base_architecture(args)

        # if not hasattr(args, 'max_positions'):
        if 'max_positions' not in args['model']:
            args['model']['max_positions'] = args['task']['tokens_per_sample']

        # encoder_params = {}
        # q_encoder = cls.make_encoder(args, task.source_dictionary)
        # k_encoder = cls.make_encoder(args, task.source_dictionary)
        if args['model']['encoder_type'] == "transformer":
            pass

        elif args['model']['encoder_type'] == "lstm":
            encoder_q = CodeEncoderLSTM(args, dictionary=task.source_dictionary,
                # n_tokens=n_tokens,
                embed_dim=args['model']['encoder_embed_dim_q'],
                hidden_size=args['model']['encoder_hidden_size_q'],
                # d_rep=d_rep,
                num_layers=args['model']['encoder_layers_q'],
                # dropout=dropout,
                # padding_idx=padding_idx,
                project='hidden' # 'hidden'
            )

            encoder_k = CodeEncoderLSTM(args, dictionary=task.source_dictionary,
                                        # n_tokens=n_tokens,
                                        embed_dim=args['model']['encoder_embed_dim_k'],
                                        hidden_size=args['model']['encoder_hidden_size_k'],
                                        # d_rep=d_rep,
                                        num_layers=args['model']['encoder_layers_k'],
                                        # dropout=dropout,
                                        # padding_idx=padding_idx,
                                        project='hidden'  #
                                        )

        return cls(args, encoder_q, encoder_k)

    # def make_encoder(self, args, dictionary): #, d_model, d_rep, pad_id=None, encoder_type="transformer", lstm_project_mode="hidden", n_encoder_layers=6, dropout=0.1, **kwargs
    #     if args['model']['encoder_type'] == "transformer":
    #         # return CodeEncoder(dictionary, project=True, pad_id=pad_id, d_model=d_model, d_rep=d_rep,
    #         #                    n_encoder_layers=n_encoder_layers, **kwargs)
    #         pass
    #     elif args['model']['encoder_type'] == "lstm":
    #         return CodeEncoderLSTM(args, dictionary=dictionary,
    #             # n_tokens=n_tokens,
    #             embed_dim=args['model']['encoder_embed_dim'],
    #             hidden_size=args['model']['encoder_hidden_size'],
    #             d_rep=d_rep,
    #             n_encoder_layers=n_encoder_layers,
    #             # dropout=dropout,
    #             padding_idx=padding_idx,
    #             project=lstm_project_mode
    #         )

        # def __init__(
        #         self,
        #         dictionary,
        #         # Deeptyper set d_model to 200
        #         embed_dim=512,
        #         hidden_size=512,
        #         d_rep=256,
        #         num_layers=2,
        #         dropout=0.1,
        #         padding_idx=None,
        #         project=False,
        #         max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS
        #
        # ):
        # else:
        #     raise ValueError

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def embed(self, img):
        return self.encoder_q(img)

    def forward(self, tokens_q, tokens_k, lengths_q, lengths_k):
        """
        Input:
            tokens_q: a batch of query images
            tokens_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        output_q = self.encoder_q(tokens_q, lengths_q)  # queries: NxC
        q = output_q['encoder_out'][0]
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            # tokens_k, idx_unshuffle = self._batch_shuffle_ddp(tokens_k)

            output_k = self.encoder_k(tokens_k, lengths_k)  # keys: NxC
            k = output_k['encoder_out'][0]
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        # q=k
        l_pos = torch.einsum("nc,nc->n", *[q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", *[q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device=logits.device)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

    def forward_(self, src_tokens, features_only=False, return_all_hiddens=False, classification_head_name=None,
                 **kwargs):
        if classification_head_name is not None:
            features_only = True

        x, extra = self.decoder(src_tokens, features_only, return_all_hiddens, **kwargs)

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
        return x, extra

    # def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
    #     """Register a classification head."""
    #     if name in self.classification_heads:
    #         prev_num_classes = self.classification_heads[name].out_proj.out_features
    #         prev_inner_dim = self.classification_heads[name].dense.out_features
    #         if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
    #             LOGGER.warning(
    #                 're-registering head "{}" with num_classes {} (prev: {}) '
    #                 'and inner_dim {} (prev: {})'.format(
    #                     name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
    #                 )
    #             )
    #     self.classification_heads[name] = RobertaClassificationHead(
    #         self.args['model']['encoder_embed_dim'],
    #         inner_dim or self.args['model']['encoder_embed_dim'],
    #         num_classes,
    #         self.args['task']['pooler_activation_fn'],
    #         self.args['model']['pooler_dropout'],
    #     )
    #
    # @property
    # def supported_targets(self):
    #     return {'self'}

    @classmethod
    def from_pretrained(cls, model_name_or_path, checkpoint_file='model.pt', data_name_or_path='.', bpe='gpt2',
                        **kwargs):
        from ncc.utils import hub_utils
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return RobertaHubInterface(x['args'], x['task'], x['models'][0])

    # def upgrade_state_dict_named(self, state_dict, name):
    #     super().upgrade_state_dict_named(state_dict, name)
    # 
    #     prefix = name + '.' if name != '' else ''
    #     current_head_names = [] if not hasattr(self, 'classification_heads') else \
    #         self.classification_heads.keys()
    # 
    #     # Handle new classification heads present in the state dict.
    #     keys_to_delete = []
    #     for k in state_dict.keys():
    #         if not k.startswith(prefix + 'classification_heads.'):
    #             continue
    # 
    #         head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
    #         num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
    #         inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)
    # 
    #         # if getattr(self.args, 'load_checkpoint_heads', False):
    #         if 'load_checkpoint_heads' in self.args['model']:
    #             if head_name not in current_head_names:
    #                 self.register_classification_head(head_name, num_classes, inner_dim)
    #         else:
    #             if head_name not in current_head_names:
    #                 LOGGER.warning(
    #                     'deleting classification head ({}) from checkpoint '
    #                     'not present in current model: {}'.format(head_name, k)
    #                 )
    #                 keys_to_delete.append(k)
    #             elif (
    #                     num_classes != self.classification_heads[head_name].out_proj.out_features
    #                     or inner_dim != self.classification_heads[head_name].dense.out_features
    #             ):
    #                 LOGGER.warning(
    #                     'deleting classification head ({}) from checkpoint '
    #                     'with different dimensions than current model: {}'.format(head_name, k)
    #                 )
    #                 keys_to_delete.append(k)
    #     for k in keys_to_delete:
    #         del state_dict[k]
    # 
    #     # Copy any newly-added classification heads into the state dict
    #     # with their current weights.
    #     if hasattr(self, 'classification_heads'):
    #         cur_state = self.classification_heads.state_dict()
    #         for k, v in cur_state.items():
    #             if prefix + 'classification_heads.' + k not in state_dict:
    #                 LOGGER.info('Overwriting ' + prefix + 'classification_heads.' + k)
    #                 state_dict[prefix + 'classification_heads.' + k] = v
    # 
    # def upgrade_state_dict_named_for_summarization(self, state_dict, name):
    #     super().upgrade_state_dict_named(state_dict, name)
    # 
    #     prefix = name + '.' if name != '' else ''
    #     current_head_names = [] if not hasattr(self, 'classification_heads') else \
    #         self.classification_heads.keys()
    # 
    #     # Handle new classification heads present in the state dict.
    #     keys_to_delete = []
    #     for k in state_dict.keys():
    #         if not k.startswith(prefix + 'classification_heads.'):
    #             continue
    # 
    #         head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
    #         num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
    #         inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)
    # 
    #         # if getattr(self.args, 'load_checkpoint_heads', False):
    #         if 'load_checkpoint_heads' in self.args['model']:
    #             if head_name not in current_head_names:
    #                 self.register_classification_head(head_name, num_classes, inner_dim)
    #         else:
    #             if head_name not in current_head_names:
    #                 LOGGER.warning(
    #                     'deleting classification head ({}) from checkpoint '
    #                     'not present in current model: {}'.format(head_name, k)
    #                 )
    #                 keys_to_delete.append(k)
    #             elif (
    #                     num_classes != self.classification_heads[head_name].out_proj.out_features
    #                     or inner_dim != self.classification_heads[head_name].dense.out_features
    #             ):
    #                 LOGGER.warning(
    #                     'deleting classification head ({}) from checkpoint '
    #                     'with different dimensions than current model: {}'.format(head_name, k)
    #                 )
    #                 keys_to_delete.append(k)
    #     for k in keys_to_delete:
    #         del state_dict[k]
    # 
    #     # Copy any newly-added classification heads into the state dict
    #     # with their current weights.
    #     if hasattr(self, 'classification_heads'):
    #         cur_state = self.classification_heads.state_dict()
    #         for k, v in cur_state.items():
    #             if prefix + 'classification_heads.' + k not in state_dict:
    #                 LOGGER.info('Overwriting ' + prefix + 'classification_heads.' + k)
    #                 state_dict[prefix + 'classification_heads.' + k] = v

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    # torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    #
    # output = torch.cat(tensors_gather, dim=0)
    output = tensor
    return output
