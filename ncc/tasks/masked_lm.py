# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import numpy as np
from ncc.data import data_utils
from ncc.data.id_dataset import IdDataset
from ncc.data.mask_tokens_dataset import MaskTokensDataset
from ncc.data.nested_dictionary_dataset import NestedDictionaryDataset
from ncc.data.numel_dataset import NumelDataset
from ncc.data.num_samples_dataset import NumSamplesDataset
from ncc.data.pad_dataset import PadDataset
from ncc.data.prepend_token_dataset import PrependTokenDataset
from ncc.data.sort_dataset import SortDataset
from ncc.data.token_block_dataset import TokenBlockDataset
from ncc.tasks.fairseq_task import FairseqTask
from ncc.tasks import register_task
from ncc.data.encoders.utils import get_whole_word_mask
from ncc.utils import utils
from ncc.data.tokenizer.tokenization_roberta import RobertaTokenizer

logger = logging.getLogger(__name__)


@register_task('masked_lm')
class MaskedLMTask(FairseqTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    # @staticmethod
    # def add_args(parser):
    #     """Add task-specific arguments to the parser."""
    #     parser.add_argument('data', help='colon separated path to data directories list, \
    #                         will be iterated upon during epochs in round-robin manner')
    #     parser.add_argument('--sample-break-mode', default='complete',
    #                         choices=['none', 'complete', 'complete_doc', 'eos'],
    #                         help='If omitted or "none", fills each sample with tokens-per-sample '
    #                              'tokens. If set to "complete", splits samples only at the end '
    #                              'of sentence, but may include multiple sentences per sample. '
    #                              '"complete_doc" is similar but respects doc boundaries. '
    #                              'If set to "eos", includes only one sentence per sample.')
    #     parser.add_argument('--tokens-per-sample', default=512, type=int,
    #                         help='max number of total tokens over all segments '
    #                              'per sample for BERT dataset')
    #     parser.add_argument('--mask-prob', default=0.15, type=float,
    #                         help='probability of replacing a token with mask')
    #     parser.add_argument('--leave-unmasked-prob', default=0.1, type=float,
    #                         help='probability that a masked token is unmasked')
    #     parser.add_argument('--random-token-prob', default=0.1, type=float,
    #                         help='probability of replacing a token with a random token')
    #     parser.add_argument('--freq-weighted-replacement', default=False, action='store_true',
    #                         help='sample random replacement words based on word frequencies')
    #     parser.add_argument('--mask-whole-words', default=False, action='store_true',
    #                         help='mask whole words; you may also want to set --bpe')

    def __init__(self, args, tokenizer):
        super().__init__(args)
        # self.dictionary = dictionary
        # self.seed = args['common']['seed']

        # # add mask token
        # self.mask_idx = dictionary.add_symbol('<mask>')、
        self.tokenizer = tokenizer
        # self.mask_idx = self.tokenizer.mask_token_id


    @classmethod
    def setup_task(cls, args, **kwargs):
        # paths = utils.split_paths(args['task']['data'])
        # print('paths: ', paths)
        # assert len(paths) > 0
        # dictionary = Dictionary.load(os.path.join(paths[0], 'dict.txt'))
        # logger.info('dictionary: {} types'.format(len(dictionary)))
        tokenizer = RobertaTokenizer.from_pretrained(args['dataset']['tokenizer_name'], cache_dir=args['checkpoint']['cache_dir'])
        return cls(args, tokenizer)

    def build_model(self, args, config):
        """
        Build the :class:`~fairseq.models.BaseFairseqModel` instance for this
        task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~fairseq.models.BaseFairseqModel` instance
        """
        from ncc import models
        # assert 0
        args['model']['arch'] = '{}_{}'.format(args['model']['arch'], args['common']['task'])
        return models.build_model(args, config, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = utils.split_paths(self.args['task']['data'])
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, 'wiki.{}.raw'.format(split))
        print('combine: ', combine)

        dataset = data_utils.load_indexed_dataset(
            path=split_path,
            dictionary=None,
            tokenizer=self.tokenizer,
            dataset_impl=self.args['dataset']['dataset_impl'],
            combine=combine,
        )
        if dataset is None:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

        # create continuous blocks of tokens
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args['task']['tokens_per_sample'] - 1,  # one less for <s>
            pad=self.tokenizer.pad_token_id,  # source_dictionary.pad(),
            eos=self.tokenizer.eos_token_id,  # .source_dictionary.eos(),
            break_mode=self.args['task']['sample_break_mode'],
        )
        logger.info('loaded {} blocks from: {}'.format(len(dataset), split_path))

        # # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        # dataset = PrependTokenDataset(dataset, self.tokenizer.bos_token_id) # .source_dictionary.bos()
        #
        # # create masked input and targets
        mask_whole_words = get_whole_word_mask(self.args, self.source_dictionary) \
            if self.args['task']['mask_whole_words'] else None
        #
        src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
            dataset,
            # self.source_dictionary,
            tokenizer=self.tokenizer,
            pad_idx=self.tokenizer.pad_token_id, # .source_dictionary.pad(),
            mask_idx=self.tokenizer.mask_token_id, #self.mask_idx,
            seed=self.args['common']['seed'],
            mask_prob=self.args['task']['mask_prob'],
            leave_unmasked_prob=self.args['task']['leave_unmasked_prob'],
            random_token_prob=self.args['task']['random_token_prob'],
            freq_weighted_replacement=self.args['task']['freq_weighted_replacement'],
            mask_whole_words=mask_whole_words,
        )

        with data_utils.numpy_seed(self.args['common']['seed'] + epoch):
            shuffle = np.random.permutation(len(src_dataset))

        self.datasets[split] = SortDataset(
            NestedDictionaryDataset(
                {
                    'id': IdDataset(),
                    'net_input': {
                        'src_tokens': PadDataset(
                            src_dataset,
                            pad_idx=self.tokenizer.pad_token_id, #source_dictionary.pad(),
                            left_pad=False,
                        ),
                        'src_lengths': NumelDataset(src_dataset, reduce=False),
                    },
                    'target': PadDataset(
                        tgt_dataset,
                        pad_idx=self.tokenizer.pad_token_id, #self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                    'nsentences': NumSamplesDataset(),
                    'ntokens': NumelDataset(src_dataset, reduce=True),
                },
                sizes=[src_dataset.sizes],
            ),
            sort_order=[
                shuffle,
                src_dataset.sizes,
            ],
        )

    def load_dataset_(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args['task']['data'])
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.source_dictionary,
            self.args['dataset']['dataset_impl'],
            combine=combine,
        )
        if dataset is None:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

        # create continuous blocks of tokens
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args['task']['tokens_per_sample'] - 1,  # one less for <s>
            pad=self.source_dictionary.pad(),
            eos=self.source_dictionary.eos(),
            break_mode=self.args['task']['sample_break_mode'],
        )
        logger.info('loaded {} blocks from: {}'.format(len(dataset), split_path))

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())

        # create masked input and targets
        mask_whole_words = get_whole_word_mask(self.args, self.source_dictionary) \
            if self.args['task']['mask_whole_words'] else None

        src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
            dataset,
            self.source_dictionary,
            pad_idx=self.source_dictionary.pad(),
            mask_idx=self.mask_idx,
            seed=self.args['common']['seed'],
            mask_prob=self.args['task']['mask_prob'],
            leave_unmasked_prob=self.args['task']['leave_unmasked_prob'],
            random_token_prob=self.args['task']['random_token_prob'],
            freq_weighted_replacement=self.args['task']['freq_weighted_replacement'],
            mask_whole_words=mask_whole_words,
        )

        with data_utils.numpy_seed(self.args['common']['seed'] + epoch):
            shuffle = np.random.permutation(len(src_dataset))

        self.datasets[split] = SortDataset(
            NestedDictionaryDataset(
                {
                    'id': IdDataset(),
                    'net_input': {
                        'src_tokens': PadDataset(
                            src_dataset,
                            pad_idx=self.source_dictionary.pad(),
                            left_pad=False,
                        ),
                        'src_lengths': NumelDataset(src_dataset, reduce=False),
                    },
                    'target': PadDataset(
                        tgt_dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                    'nsentences': NumSamplesDataset(),
                    'ntokens': NumelDataset(src_dataset, reduce=True),
                },
                sizes=[src_dataset.sizes],
            ),
            sort_order=[
                shuffle,
                src_dataset.sizes,
            ],
        )

    # def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
    #     src_dataset = PadDataset(
    #         TokenBlockDataset(
    #             src_tokens,
    #             src_lengths,
    #             self.args['task']['tokens_per_sample'] - 1,  # one less for <s>
    #             pad=self.source_dictionary.pad(),
    #             eos=self.source_dictionary.eos(),
    #             break_mode='eos',
    #         ),
    #         pad_idx=self.source_dictionary.pad(),
    #         left_pad=False,
    #     )
    #     src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.bos())
    #     src_dataset = NestedDictionaryDataset(
    #         {
    #             'id': IdDataset(),
    #             'net_input': {
    #                 'src_tokens': src_dataset,
    #                 'src_lengths': NumelDataset(src_dataset, reduce=False),
    #             },
    #         },
    #         sizes=src_lengths,
    #     )
    #     if sort:
    #         src_dataset = SortDataset(src_dataset, sort_order=[src_lengths])
    #     return src_dataset
    #
    @property
    def source_dictionary(self):
        return self.tokenizer.encoder

    @property
    def target_dictionary(self):
        return self.tokenizer.encoder
