# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np

# from fairseq.data import (
#     data_utils,
#     Dictionary,
#     IdDataset,
#     MaskTokensDataset,
#     NestedDictionaryDataset,
#     NumelDataset,
#     NumSamplesDataset,
#     PadDataset,
#     PrependTokenDataset,
#     SortDataset,
#     TokenBlockDataset,
# )
from ncc.data import data_utils
from ncc.data.dictionary import Dictionary
from ncc.data.id_dataset import IdDataset
from ncc.data.mask_tokens_dataset import MaskTokensDataset
from ncc.data.nested_dictionary_dataset import NestedDictionaryDataset
from ncc.data.numel_dataset import NumelDataset
from ncc.data.num_samples_dataset import NumSamplesDataset
from ncc.data.pad_dataset import PadDataset
from ncc.data.prepend_token_dataset import PrependTokenDataset
from ncc.data.sort_dataset import SortDataset
from ncc.data.token_block_dataset import TokenBlockDataset

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from pathlib import Path

from .fairseq_task import FairseqTask
from . import register_task
from ncc.data.encoder.utils import get_whole_word_mask
from ncc.utils import utils
# from ncc.models.bert.modeling_bert import BertForMaskedLM
# from ncc.models.bert.modeling_roberta import RobertaForMaskedLM
# from ncc.config.bert.configuration_bert import BertConfig
# from ncc.config.bert.configuration_roberta import RobertaConfig
# from ncc.data.tokenizer.tokenization_bert import BertTokenizer
from ncc.data.tokenizer.tokenization_roberta import RobertaTokenizer
# from ncc.utils.modeling_utils import PreTrainedModel
# from ncc.utils.tokenization_utils import PreTrainedTokenizer

logger = logging.getLogger(__name__)


@register_task('summarization')
class SummarizationTask(FairseqTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        # self.dictionary = dictionary
        # self.seed = args['common']['seed']

        # # add mask token
        # self.mask_idx = dictionary.add_symbol('<mask>')、
        # self.tokenizer = tokenizer
        # self.mask_idx = self.tokenizer.mask_token_id
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = utils.split_paths(args['task']['data'])
        print('paths: ', paths)
        # assert len(paths) > 0
        # dictionary = Dictionary.load(os.path.join(paths[0], 'dict.txt'))
        # logger.info('dictionary: {} types'.format(len(dictionary)))
        # tokenizer = RobertaTokenizer.from_pretrained(args['dataset']['tokenizer_name'], cache_dir=args['checkpoint']['cache_dir'])
        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info('[{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        logger.info('[{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

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

    @property
    def source_dictionary(self):
        return self.tokenizer.encoder

    @property
    def target_dictionary(self):
        return self.tokenizer.encoder
