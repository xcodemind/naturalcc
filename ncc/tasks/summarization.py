# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import *
import os
import json
import itertools
from argparse import Namespace
from ncc import LOGGER
from ncc.tasks.fairseq_task import FairseqTask
from ncc.tasks import register_task
from ncc.utils import utils
from ncc.data import encoders
from ncc.data import indexed_dataset
from ncc.data import data_utils
from ncc.data.append_token_dataset import AppendTokenDataset
from ncc.data.truncate_dataset import TruncateDataset
from ncc.data.strip_token_dataset import StripTokenDataset
from ncc.data.concat_dataset import ConcatDataset
from ncc.data.prepend_token_dataset import PrependTokenDataset
from ncc.data.language_pair_dataset import LanguagePairDataset
from ncc.data.dictionary import Dictionary
from ncc.utils import tokenizer  # , utils # metrics, search,


def load_langpair_dataset(
        data_path, split,
        src, src_dict,
        tgt, tgt_dict,
        combine, dataset_impl, upsample_primary,
        left_pad_source, left_pad_target, max_source_positions,
        max_target_positions, prepend_bos=False, load_alignments=False,
        truncate_source=False, append_source_id=False
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}'.format(split, src))  # -{}.{} , tgt, lang
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.'.format(split_k))  # {}-{}. , src, tgt
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, '{}.'.format(split_k))  # {}-{}. , tgt, src

        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))
        print('prefix + src: ', prefix + src)
        src_dataset = data_utils.load_indexed_dataset(prefix + src, 'text', src_dict, dataset_impl)
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(prefix + tgt, 'text', tgt_dict, dataset_impl)
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        LOGGER.info('{} {} {}-{} {} examples'.format(
            data_path, split_k, src, tgt, len(src_datasets[-1])
        ))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(src_dataset, src_dict.index('[{}]'.format(src)))
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(tgt_dataset, tgt_dict.index('[{}]'.format(tgt)))
        eos = tgt_dict.index('[{}]'.format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, '{}.align.{}-{}'.format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(align_path, None, dataset_impl)

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset_sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        align_dataset=align_dataset, eos=eos
    )


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
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        # options.eval_bool
        args['task']['left_pad_source'] = bool(args['task']['left_pad_source'])
        args['task']['left_pad_target'] = bool(args['task']['left_pad_target'])

        paths = utils.split_paths(args['task']['data'])
        assert len(paths) > 0
        # find language pair automatically
        if args['task']['source_lang'] is None or args['task']['target_lang'] is None:
            args['task']['source_lang'], args['task']['target_lang'] = data_utils.infer_language_pair(paths[0])
        if args['task']['source_lang'] is None or args['task']['target_lang'] is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args['task']['source_lang'])))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args['task']['target_lang'])))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        LOGGER.info('[{}] dictionary: {} types'.format(args['task']['source_lang'], len(src_dict)))
        LOGGER.info('[{}] dictionary: {} types'.format(args['task']['target_lang'], len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

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

        self.datasets[split] = load_langpair_dataset(
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

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def build_model(self, args):
        model = super().build_model(args)
        if getattr(args, 'eval_bleu', False):
            assert getattr(args, 'eval_bleu_detok', None) is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            detok_args = json.loads(getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
            self.tokenizer = encoders.build_tokenizer(Namespace(
                tokenizer=getattr(args, 'eval_bleu_detok', None),
                **detok_args
            ))

            gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
            self.sequence_generator = self.build_generator([model], Namespace(**gen_args))
        return model

    @classmethod
    def build_dictionary(
            cls, filenames: List, modality: str, tokenize_func: Any,
            eos_word: Optional[str] = None, workers: int = 1, threshold: int = -1, nwords: int = -1,
            padding_factor: int = 8,
    ) -> Union[List[Dictionary], Dictionary]:
        """
        Build the dictionary

        Args:
            filenames (list): list of filenames
            workers (int): number of concurrent workers
            threshold (int): defines the minimum word count
            nwords (int): defines the total number of words in the final dictionary,
                including special symbols
            padding_factor (int): can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        dict_num = tokenizer.tokinzer_returns(tokenize_func)
        dicts = [Dictionary() for _ in range(dict_num)]
        for filename in filenames:
            Dictionary.add_file_to_dictionary(
                filename, dicts, tokenize_func, eos_word, workers
            )
        for dict in dicts:
            dict.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        if dict_num > 1:
            return dicts
        else:
            return dicts[0]

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
