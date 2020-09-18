# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from ncc.data.tools import data_utils
from ncc.data.id_dataset import IdDataset
from ncc.data.contracode.contra_mask_tokens_dataset import ContraMaskTokensDataset
from ncc.data.nested_dictionary_dataset import NestedDictionaryDataset
from ncc.data.wrappers.numel_dataset import NumelDataset
from ncc.data.num_samples_dataset import NumSamplesDataset
from ncc.data.wrappers.pad_dataset import PadDataset
from ncc.tasks.fairseq_task import FairseqTask
from ncc.tasks import register_task
from ncc.utils import utils
from ncc import LOGGER
from ncc.data import constants
from ncc.data.dictionary import Dictionary
from ncc.data.encoders.utils import get_whole_word_mask
from ncc.data.contracode.contracode_dataset import ContraCodeDataset


def load_masked_code_dataset(args, epoch, data_path, split, source_dictionary, combine,):
    split_path = os.path.join(data_path, '{}.sp.json'.format(split)) # '{}.code'.format(split)
    dataset = data_utils.load_indexed_dataset(
        path=split_path,
        modality='javascript',
        dictionary=source_dictionary,
        # tokenizer=sp.EncodeAsPieces,
        dataset_impl=args['dataset']['dataset_impl'],
        combine=combine,
    )
    if dataset is None:
        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

    dataset = ContraCodeDataset(dataset, dataset.sizes, source_dictionary, program_mode='identity', shuffle=False,)
    # create masked input and targets
    mask_whole_words = get_whole_word_mask(args, source_dictionary) if args['task']['mask_whole_words'] else None

    src_dataset, tgt_dataset = ContraMaskTokensDataset.apply_mask(
        dataset,
        source_dictionary,
        pad_idx=source_dictionary.pad(),
        mask_idx=source_dictionary.index(constants.MASK),
        seed=args['common']['seed'],
        mask_prob=args['task']['mask_prob'],
        leave_unmasked_prob=args['task']['leave_unmasked_prob'],
        random_token_prob=args['task']['random_token_prob'],
        freq_weighted_replacement=args['task']['freq_weighted_replacement'],
        mask_whole_words=mask_whole_words,
    )
    # print(src_dataset.__getitem__(0))
    # return ContraCodeDataset(
    #     dataset, dataset.sizes, source_dictionary, program_mode='contrastive',
    #     shuffle=False,
    # )
    # with data_utils.numpy_seed(args['common']['seed'] + epoch):
    #     shuffle = np.random.permutation(len(src_dataset))
    dataset = NestedDictionaryDataset(
        {
            'id': IdDataset(),
            'net_input': {
                'src_tokens': PadDataset(
                    src_dataset,
                    pad_idx=source_dictionary.pad(),
                    left_pad=False,
                ),
                'src_lengths': NumelDataset(src_dataset, reduce=False),
            },
            'target': PadDataset(
                tgt_dataset,
                pad_idx=source_dictionary.pad(),
                left_pad=False,
            ),
            'nsentences': NumSamplesDataset(),
            'ntokens': NumelDataset(src_dataset, reduce=True),
        },
        sizes=[src_dataset.sizes],
    )
    return dataset


@register_task('contracode_mlm')
class ContraCodeMLM(FairseqTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args['common']['seed']
        # self.sp = sp
        # add mask token
        # self.mask_idx = self.dictionary.add_symbol(constants.MASK)

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        if filename.endswith('.txt'):
            dictionary = Dictionary(
                extra_special_symbols=[constants.CLS, constants.SEP, constants.MASK,
                                       constants.EOL, constants.URL])
            dictionary.add_from_file(filename)
        else:
            dictionary = Dictionary(extra_special_symbols=[constants.CLS, constants.SEP, constants.MASK,
                                       constants.EOL, constants.URL]).add_from_json_file(filename)
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = utils.split_paths(args['task']['data'])
        assert len(paths) > 0
        dictionary = cls.load_dictionary(os.path.join(paths[0], 'csnjs_8k_9995p_unigram_url.dict.txt'))
        LOGGER.info('dictionary: {} types'.format(len(dictionary)))
        # sp = spm.SentencePieceProcessor()
        # sp.Load(os.path.join(paths[0], 'csnjs_8k_9995p_unigram_url.model'))

        return cls(args, dictionary)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args['task']['data'])
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        self.datasets[split] = load_masked_code_dataset(self.args, epoch, data_path, split, self.source_dictionary, combine) #, self.sp,

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary
