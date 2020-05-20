# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
from ncc.data.dictionary import Dictionary
from ncc import LOGGER
from ncc.data.append_token_dataset import AppendTokenDataset
from ncc.data.truncate_dataset import TruncateDataset
from ncc.data.strip_token_dataset import StripTokenDataset
from ncc.data.concat_dataset import ConcatDataset
from ncc.data.prepend_token_dataset import PrependTokenDataset
from ncc.data.mask_code_docstring_pair_dataset import MaskCodeDocstringPairDataset
from ncc.data import constants


# dataset = load_masked_code_docstring_dataset(
#         data_path, split, args['task']['source_lang'], src_dict, args['task']['target_lang'], tgt_dict,
#         combine=combine, dataset_impl=args['dataset']['dataset_impl'],
#         upsample_primary=args['task']['upsample_primary'],
#         left_pad_source=args['task']['left_pad_source'],
#         left_pad_target=args['task']['left_pad_target'],
#         max_source_positions=args['task']['max_source_positions'],
#         max_target_positions=args['task']['max_target_positions'],
#          =args['task']['load_alignments'],
#         truncate_source=args['task']['truncate_source'],
#     )
def load_masked_code_docstring_dataset(
        data_path, split,
        src, src_dict,
        tgt, tgt_dict,
        combine, dataset_impl, upsample_primary,
        left_pad_source, left_pad_target, max_source_positions,
        max_target_positions, prepend_bos=False, load_alignments=False,
        truncate_source=False, append_source_id=False):

    src_path = os.path.join(data_path, '{}.code.bpe'.format(split))
    target_path = os.path.join(data_path, '{}.docstring.bpe'.format(split))

    # src_dataset
    src_dataset = data_utils.load_indexed_dataset(src_path, 'text', src_dict, dataset_impl)
    # if truncate_source:
    #     src_dataset = AppendTokenDataset(
    #         TruncateDataset(
    #             StripTokenDataset(src_dataset, src_dict.eos()),
    #             max_source_positions - 1,
    #         ),
    #         src_dict.eos(),
    #     )

    # tgt_dataset
    tgt_dataset = data_utils.load_indexed_dataset(target_path, 'text', tgt_dict, dataset_impl)

    eos = None
    align_dataset = None
    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None

    return MaskCodeDocstringPairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset_sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        align_dataset=align_dataset, eos=eos,
        skipgram_prb=0.0,
        skipgram_size=0.0,
    )
    # dataset = data_utils.load_indexed_dataset(
    #     split_path,
    #     dictionary=self.source_dictionary,
    #     self.args['dataset']['dataset_impl'],
    #     combine=combine,
    # )
    # dataset = data_utils.load_indexed_dataset(
    #     path=split_path,
    #     modality='text',
    #     dictionary=self.source_dictionary,
    #     tokenizer=None,  # the tokenizer is only for huggingface
    #     dataset_impl=self.args['dataset']['dataset_impl'],
    #     combine=combine,
    # )
    # if dataset is None:
    #     raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))
    #
    # # create continuous blocks of tokens
    # dataset = TokenBlockDataset(
    #     dataset,
    #     dataset.sizes,
    #     self.args['task']['tokens_per_sample'] - 1,  # one less for <s>
    #     pad=self.source_dictionary.pad(),
    #     eos=self.source_dictionary.eos(),
    #     break_mode=self.args['task']['sample_break_mode'],
    # )
    # LOGGER.info('loaded {} blocks from: {}'.format(len(dataset), split_path))
    #
    # # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
    # dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())
    #
    # # create masked input and targets
    # mask_whole_words = get_whole_word_mask(self.args, self.source_dictionary) \
    #     if self.args['task']['mask_whole_words'] else None
    #
    # src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
    #     dataset,
    #     self.source_dictionary,
    #     pad_idx=self.source_dictionary.pad(),
    #     mask_idx=self.mask_idx,
    #     seed=self.args['common']['seed'],
    #     mask_prob=self.args['task']['mask_prob'],
    #     leave_unmasked_prob=self.args['task']['leave_unmasked_prob'],
    #     random_token_prob=self.args['task']['random_token_prob'],
    #     freq_weighted_replacement=self.args['task']['freq_weighted_replacement'],
    #     mask_whole_words=mask_whole_words,
    # )
    #
    # with data_utils.numpy_seed(self.args['common']['seed'] + epoch):
    #     shuffle = np.random.permutation(len(src_dataset))
    #
    # self.datasets[split] = SortDataset(
    #     NestedDictionaryDataset(
    #         {
    #             'id': IdDataset(),
    #             'net_input': {
    #                 'src_tokens': PadDataset(
    #                     src_dataset,
    #                     pad_idx=self.source_dictionary.pad(),
    #                     left_pad=False,
    #                 ),
    #                 'src_lengths': NumelDataset(src_dataset, reduce=False),
    #             },
    #             'target': PadDataset(
    #                 tgt_dataset,
    #                 pad_idx=self.source_dictionary.pad(),
    #                 left_pad=False,
    #             ),
    #             'nsentences': NumSamplesDataset(),
    #             'ntokens': NumelDataset(src_dataset, reduce=True),
    #         },
    #         sizes=[src_dataset.sizes],
    #     ),
    #     sort_order=[
    #         shuffle,
    #         src_dataset.sizes,
    #     ],
    # )


@register_task('masked_code_docstring')
class MaskedCodeDocstringTask(FairseqTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        # self.dictionary = dictionary
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.seed = args['common']['seed']

        # add mask token
        # self.mask_idx = dictionary.add_symbol('<mask>')

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = utils.split_paths(args['task']['data'])
        assert len(paths) > 0
        # dictionary = Dictionary.load(os.path.join(paths[0], 'dict.code.txt'))
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], 'dict.{}.txt'.format(args['task']['source_lang'])))  # args['task']['source_lang']
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], 'dict.{}.txt'.format(args['task']['target_lang'])))

        src_dict.add_symbol(constants.S_SEP)
        src_dict.add_symbol(constants.S2S_SEP)
        src_dict.add_symbol(constants.CLS)
        src_dict.add_symbol(constants.T_MASK)
        src_dict.add_symbol(constants.SEP)

        tgt_dict.add_symbol(constants.S2S_BOS)
        tgt_dict.add_symbol(constants.T_MASK)
        tgt_dict.add_symbol(constants.SEP)
        print('<T_MASK> id is', src_dict.index('<T_MASK>'))
        print('<T_MASK> id is', tgt_dict.index('<T_MASK>'))

        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()

        return cls(args, src_dict, tgt_dict)

    def load_dataset_raw(self, split, epoch=1, combine=False, **kwargs):
        paths = utils.split_paths(self.args['task']['data'])
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, 'wiki.{}.raw'.format(split))
        # print('combine: ', combine)

        # dataset = data_utils.load_indexed_dataset(
        #     path=split_path,
        #     dictionary=None,
        #     tokenizer=self.tokenizer,
        #     dataset_impl=self.args['dataset']['dataset_impl'],
        #     combine=combine,
        # )
        dataset = data_utils.load_indexed_dataset(
            path=split_path,
            modality='text',
            dictionary=self.source_dictionary,
            tokenizer=None,  # the tokenizer is only for huggingface
            dataset_impl=self.args['dataset']['dataset_impl'],
            combine=combine,
        )
        if dataset is None:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

        # create continuous blocks of tokens
        # dataset = TokenBlockDataset(
        #     dataset,
        #     dataset.sizes,
        #     self.args['task']['tokens_per_sample'] - 1,  # one less for <s>
        #     pad=self.tokenizer.pad_token_id,  # source_dictionary.pad(),
        #     eos=self.tokenizer.eos_token_id,  # .source_dictionary.eos(),
        #     break_mode=self.args['task']['sample_break_mode'],
        # )
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args['task']['tokens_per_sample'] - 1,  # one less for <s>
            pad=self.source_dictionary.pad(),
            eos=self.source_dictionary.eos(),
            break_mode=self.args['task']['sample_break_mode'],
        )
        LOGGER.info('loaded {} blocks from: {}'.format(len(dataset), split_path))

        # # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        # dataset = PrependTokenDataset(dataset, self.tokenizer.bos_token_id) # .source_dictionary.bos()
        #
        # # create masked input and targets
        mask_whole_words = get_whole_word_mask(self.args, self.source_dictionary) \
            if self.args['task']['mask_whole_words'] else None
        #
        # src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
        #     dataset,
        #     # self.source_dictionary,
        #     tokenizer=self.tokenizer,
        #     pad_idx=self.tokenizer.pad_token_id, # .source_dictionary.pad(),
        #     mask_idx=self.tokenizer.mask_token_id, #self.mask_idx,
        #     seed=self.args['common']['seed'],
        #     mask_prob=self.args['task']['mask_prob'],
        #     leave_unmasked_prob=self.args['task']['leave_unmasked_prob'],
        #     random_token_prob=self.args['task']['random_token_prob'],
        #     freq_weighted_replacement=self.args['task']['freq_weighted_replacement'],
        #     mask_whole_words=mask_whole_words,
        # )
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

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args['task']['data'])
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args['task']['source_lang'], self.args['task']['target_lang']

        self.datasets[split] = load_masked_code_docstring_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args['dataset']['dataset_impl'],
            upsample_primary=self.args['task']['upsample_primary'],
            left_pad_source=self.args['task']['left_pad_source'],
            left_pad_target=self.args['task']['left_pad_target'],
            max_source_positions=self.args['task']['max_source_positions'],
            max_target_positions=self.args['task']['max_target_positions'],
            load_alignments=self.args['task']['load_alignments'],
            truncate_source=self.args['task']['truncate_source'],
        )


    def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
        src_dataset = PadDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                self.args['task']['tokens_per_sample'] - 1,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode='eos',
            ),
            pad_idx=self.source_dictionary.pad(),
            left_pad=False,
        )
        src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.bos())
        src_dataset = NestedDictionaryDataset(
            {
                'id': IdDataset(),
                'net_input': {
                    'src_tokens': src_dataset,
                    'src_lengths': NumelDataset(src_dataset, reduce=False),
                },
            },
            sizes=src_lengths,
        )
        if sort:
            src_dataset = SortDataset(src_dataset, sort_order=[src_lengths])
        return src_dataset

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.tgt_dict