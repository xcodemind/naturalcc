# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
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
from ncc.data.tools import data_utils
from ncc.data.wrappers.append_token_dataset import AppendTokenDataset
from ncc.data.wrappers.truncate_dataset import TruncateDataset
from ncc.data.wrappers.strip_token_dataset import StripTokenDataset
from ncc.data.concat_dataset import ConcatDataset
from ncc.data.wrappers.prepend_token_dataset import PrependTokenDataset
from ncc.data.summarization.multi_language_pair_dataset import MultiLanguagePairDataset
from typing import Dict
from collections import OrderedDict
from ncc.data.dictionary import Dictionary
from ncc.utils import tokenizer # , utils # metrics, search,


def load_multimodalpair_dataset(
    data_path, split,
    src_modalities, src_dicts,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions,
    max_target_positions, prepend_bos=False, load_alignments=False,
    truncate_source=False, append_source_id=False
):
    # TODO: add modality argument
    def split_exists(split, modality, data_path):
        # filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        filename = os.path.join(data_path, '{}.{}'.format(split, modality))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    # code_datasets = []
    # path_datasets = []
    # ast_datasets = []
    src_datasets = {modality: [] for modality in src_modalities}
    src_dataset = {modality: None for modality in src_modalities}
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        # if split_exists(split_k, src, tgt, src, data_path):
        #     prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        # elif split_exists(split_k, tgt, src, src, data_path):
        #     prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))

        if 'code' in src_modalities:
            if split_exists(split_k, 'code', data_path):
                prefix = os.path.join(data_path, '{}.'.format(split_k))
            # elif split_exists(split_k, tgt, src, src, data_path):
            #     prefix = os.path.join(data_path, '{}.'.format(split_k))
            else:
                if k > 0:
                    break
                else:
                    raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

            code_dataset = data_utils.load_indexed_dataset(prefix + 'code', dictionary=src_dicts['code'], dataset_impl=dataset_impl)
            if truncate_source:
                code_dataset = AppendTokenDataset(
                    TruncateDataset(
                        StripTokenDataset(code_dataset, src_dicts['code'].eos()),
                        max_source_positions - 1,
                    ),
                    src_dicts['code'].eos(),
                )
            src_datasets['code'].append(code_dataset)

        # ast_datasets
        if 'bin_ast' in src_modalities:
            if split_exists(split_k, 'raw_ast', data_path):
                prefix = os.path.join(data_path, '{}.'.format(split_k))
            # elif split_exists(split_k, tgt, src, src, data_path):
            #     prefix = os.path.join(data_path, '{}.'.format(split_k))
            else:
                if k > 0:
                    break
                else:
                    raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))
            ast_dataset = data_utils.load_indexed_dataset(prefix + 'raw_ast', 'ast', dictionary=src_dicts['raw_ast'], dataset_impl=dataset_impl)
            if truncate_source:
                ast_dataset = AppendTokenDataset(
                    TruncateDataset(
                        StripTokenDataset(ast_dataset, src_dicts['raw_ast'].eos()),
                        max_source_positions - 1,
                    ),
                    src_dicts['raw_ast'].eos(),
                )
            src_datasets['ast'].append(ast_dataset)

        # path_datasets
        if 'path' in src_modalities:
            if split_exists(split_k, 'path', data_path):
                prefix = os.path.join(data_path, '{}.'.format(split_k))
            # elif split_exists(split_k, tgt, src, src, data_path):
            #     prefix = os.path.join(data_path, '{}.'.format(split_k))
            else:
                if k > 0:
                    break
                else:
                    raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))
            path_dataset = data_utils.load_indexed_dataset(prefix + 'path', 'path', dictionary=src_dicts['path'], dataset_impl=dataset_impl)
            if truncate_source:
                path_dataset = AppendTokenDataset(
                    TruncateDataset(
                        StripTokenDataset(path_dataset, src_dicts['path'].eos()),
                        max_source_positions - 1,
                    ),
                    src_dicts['path'].eos(),
                )
            src_datasets['path'].append(path_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(prefix + tgt, dictionary=tgt_dict, dataset_impl=dataset_impl)
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        # LOGGER.info('{} {} {}-{} {} examples'.format(
        #     data_path, split_k, src, tgt, len(code_datasets[-1])
        # ))

        if not combine:
            break

    for modality in src_modalities:
        assert len(src_datasets[modality]) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(tgt_datasets) == 1:
        # code_dataset = code_datasets[0]
        # path_dataset = path_datasets[0]
        # ast_dataset = ast_datasets[0]
        for modality in src_modalities:
            src_dataset[modality] = src_datasets[modality][0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        for modality in src_modalities:
            sample_ratios = [1] * len(src_datasets[modality])
            sample_ratios[0] = upsample_primary
            src_dataset[modality] = ConcatDataset(src_datasets[modality], sample_ratios)
            # path_dataset = ConcatDataset(path_datasets, sample_ratios)
            # ast_dataset = ConcatDataset(ast_datasets, sample_ratios)
            if len(tgt_datasets) > 0:
                tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
            else:
                tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dicts['code'], "bos_index") and hasattr(tgt_dict, "bos_index")
        for modality in src_modalities:
            src_dataset[modality] = PrependTokenDataset(src_dataset[modality], src_dicts[modality].bos())
            # code_dataset = PrependTokenDataset(code_dataset, src_dict['code'].bos())
            # path_dataset = PrependTokenDataset(path_dataset, src_dict['ast'].bos())
            # ast_dataset = PrependTokenDataset(ast_dataset, src_dict['path'].bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        for modality in src_modalities:
            src_dataset[modality] = AppendTokenDataset(src_dataset[modality], src_dicts[modality].index('[{}]'.format(modality)))
        # code_dataset = AppendTokenDataset(code_dataset, src_dict['code'].index('[{}]'.format(src)))
        # path_dataset = AppendTokenDataset(path_dataset, src_dict['ast'].index('[{}]'.format(src)))
        # ast_dataset = AppendTokenDataset(ast_dataset, src_dict['path'].index('[{}]'.format(src)))
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(tgt_dataset, tgt_dict.index('[{}]'.format(tgt)))
        eos = tgt_dict.index('[{}]'.format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, '{}.align.{}-{}'.format(split, src_modalities, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(align_path, None, dataset_impl)

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None

    src_dataset_sizes = {modality: src_dataset[modality].sizes for modality in src_modalities}
    return MultiLanguagePairDataset(
        src_dataset, src_dataset_sizes, src_dicts,
        src_modalities,
        tgt_dataset, tgt_dataset_sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        align_dataset=align_dataset, eos=eos
    )


@register_task('mm_summarization')
class MMSummarizationTask(FairseqTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    def __init__(self, args, src_dicts: OrderedDict, tgt_dict):
        super().__init__(args)
        # self.dictionary = dictionary
        # self.seed = args['common']['seed']

        # # add mask token
        # self.mask_idx = dictionary.add_symbol('<mask>')、
        # self.tokenizer = tokenizer
        # self.mask_idx = self.tokenizer.mask_token_id
        self.src_dicts = src_dicts
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
        # if args['task']['source_lang'] is None or args['task']['target_lang'] is None:
        #     args['task']['source_lang'], args['task']['target_lang'] = data_utils.infer_language_pair(paths[0])
        if args['task']['source_lang'] is None or args['task']['target_lang'] is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args['task']['target_lang'])))
        # src_dict = {modality: None for modality in args['task']['source_lang']}
        src_dicts = OrderedDict()
        for modality in args['task']['source_lang']:
            if modality == 'path':  # special for path modality
                dict_path_border = cls.load_dictionary(os.path.join(paths[0], 'dict.{}_border.txt'.format(modality)))  # args['task']['source_lang']
                dict_path_center = cls.load_dictionary(os.path.join(paths[0], 'dict.{}_center.txt'.format(modality)))  # args['task']['source_lang']
                src_dicts[modality] = [dict_path_border, dict_path_center]
                assert src_dicts[modality][0].pad() == src_dicts[modality][1].pad() == tgt_dict.pad()
                assert src_dicts[modality][0].eos() == src_dicts[modality][1].eos() == tgt_dict.eos()
                assert src_dicts[modality][0].unk() == src_dicts[modality][1].unk() == tgt_dict.unk()
            else:
                src_dicts[modality] = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(modality)))    #args['task']['source_lang']
                assert src_dicts[modality].pad() == tgt_dict.pad()
                assert src_dicts[modality].eos() == tgt_dict.eos()
                assert src_dicts[modality].unk() == tgt_dict.unk()
        # LOGGER.info('[{}] dictionary: {} types'.format(args['task']['source_lang'], len(src_dict)))
        LOGGER.info('[{}] dictionary: {} types'.format(args['task']['target_lang'], len(tgt_dict)))

        return cls(args, src_dicts, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args['task']['data'])
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src_modalities, tgt = self.args['task']['source_lang'], self.args['task']['target_lang']

        self.datasets[split] = load_multimodalpair_dataset(
            data_path, split, src_modalities, self.src_dicts, tgt, self.tgt_dict,
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
        return MultiLanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

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

    # TODO: rewrite this function, add reinforcement learning
    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    @classmethod
    def build_dictionary(
            cls, filenames, modality='code', workers=1, threshold=-1, nwords=-1, padding_factor=8
    ):
        """Build the dictionary

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
        if modality == 'code':
            d = Dictionary()
            for filename in filenames:
                Dictionary.add_file_to_dictionary(
                    filename, d, tokenizer.tokenize_line, workers
                )
            d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
            return d

        elif modality == 'path':
            # for border of path
            d_border = Dictionary()
            for filename in filenames:
                Dictionary.add_listfile_to_dictionary(
                    filename, d_border, 'border', tokenizer.tokenize_path_line, workers
                )
            d_border.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
            # for center of path
            d_center = Dictionary()
            for filename in filenames:
                Dictionary.add_listfile_to_dictionary(
                    filename, d_center, 'center',tokenizer.tokenize_path_line, workers
                )
            d_center.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
            return d_border, d_center

        elif modality == 'bin_ast':
            d = Dictionary()
            for filename in filenames:
                Dictionary.add_jsonfile_to_dictionary(
                    filename, d, tokenizer.tokenize_tree_line, workers
                )
            d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
            return d


    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args['task']['max_source_positions'], self.args['task']['max_target_positions'])

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dicts

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict