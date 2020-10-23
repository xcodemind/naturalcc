# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import json
import itertools
from argparse import Namespace
from ncc import LOGGER
from ncc.tasks.ncc_task import NccTask
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
from ncc.data.summarization.multi_modalities_pair_dataset import MultiModalitiesPairDataset
from typing import Dict
from collections import OrderedDict
from ncc.data.dictionary import Dictionary
from ncc.utils import tokenizer  # , utils # metrics, search,


def load_multimodalpair_dataset(
    data_path, split,
    srcs, src_dicts,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions,
    max_target_positions, prepend_bos=False, load_alignments=False,
    truncate_source=False, append_source_id=False,
    **kwargs,
):
    def split_exists(split, modality, data_path):
        filename = os.path.join(data_path, '{}.{}'.format(split, modality))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = OrderedDict()
    for modality in srcs:
        src_datasets[modality] = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        for modality in srcs:
            # infer langcode
            if split_exists(split_k, modality, data_path):
                prefix = os.path.join(data_path, '{}.'.format(split_k))  # {}-{}. , src, tgt
            elif split_exists(split_k, tgt, data_path):
                prefix = os.path.join(data_path, '{}.'.format(split_k))  # {}-{}. , tgt, src

            else:
                if k > 0:
                    break
                else:
                    raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

            src_dataset = data_utils.load_indexed_dataset(prefix + modality, modality, src_dicts[modality],
                                                          dataset_impl)
            if truncate_source:
                src_dataset = AppendTokenDataset(
                    TruncateDataset(
                        StripTokenDataset(src_dataset, src_dicts[modality].eos()),
                        max_source_positions - 1,
                    ),
                    src_dicts[modality].eos(),
                )
            src_datasets[modality].append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(prefix + tgt, dictionary=tgt_dict, dataset_impl=dataset_impl)
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        if not combine:
            break

    for modality in srcs:
        assert len(src_datasets[modality]) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(tgt_datasets) == 1:
        src_dataset = OrderedDict()
        for modality in srcs:
            src_dataset[modality] = src_datasets[modality][0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        raise NotImplementedError

    if prepend_bos:
        assert hasattr(src_dicts[modality], "bos_index") and hasattr(tgt_dict, "bos_index")
        for modality in srcs:
            src_dataset[modality] = PrependTokenDataset(src_dataset[modality], src_dicts[modality].bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        for modality in srcs:
            src_dataset[modality] = AppendTokenDataset(src_dataset[modality],
                                                       src_dicts[modality].index('[{}]'.format(modality)))
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(tgt_dataset, tgt_dict.index('[{}]'.format(tgt)))
        eos = tgt_dict.index('[{}]'.format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, '{}.align.{}-{}'.format(split, srcs, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(align_path, None, dataset_impl)

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    src_dataset_sizes = OrderedDict()
    for modality in srcs:
        src_dataset_sizes[modality] = src_dataset[modality].sizes
    return MultiModalitiesPairDataset(
        src_dataset, src_dataset_sizes, src_dicts,
        tgt_dataset, tgt_dataset_sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        align_dataset=align_dataset, eos=eos,
        num_sample=kwargs.get('num_sample', None),
    )


@register_task('mm_summarization')
class MMSummarizationTask(NccTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    def __init__(self, args, src_dicts: OrderedDict, tgt_dict):
        super().__init__(args)
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
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], '{}.dict.json'.format(args['task']['target_lang'])))
        src_dicts = OrderedDict()
        for modality in args['task']['source_lang']:
            src_dicts[modality] = cls.load_dictionary(os.path.join(paths[0], '{}.dict.json'.format(modality)))
            assert src_dicts[modality].pad() == tgt_dict.pad()
            assert src_dicts[modality].eos() == tgt_dict.eos()
            assert src_dicts[modality].unk() == tgt_dict.unk()
        LOGGER.info('[{}] dictionary: {} types'.format(
            args['task']['source_lang'], [len(src_dicts[modality]) for modality in args['task']['source_lang']])
        )
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
        srcs, tgt = self.args['task']['source_lang'], self.args['task']['target_lang']

        self.datasets[split] = load_multimodalpair_dataset(
            data_path, split, srcs, self.src_dicts, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args['dataset']['dataset_impl'],
            upsample_primary=self.args['task']['upsample_primary'],
            left_pad_source=self.args['task']['left_pad_source'],
            left_pad_target=self.args['task']['left_pad_target'],
            max_source_positions=self.args['task']['max_source_positions'],
            max_target_positions=self.args['task']['max_target_positions'],
            load_alignments=self.args['task']['load_alignments'],
            truncate_source=self.args['task']['truncate_source'],
            num_sample=self.args['task']['num_sample'],
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return MultiModalitiesPairDataset(src_tokens, src_lengths, self.source_dictionary)

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
            model (~fairseq.models.BaseNccModel): the model
            criterion (~fairseq.criterions.NccCriterion): the criterion
            optimizer (~fairseq.optim.NccOptimizer): the optimizer
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
                    filename, d_center, 'center', tokenizer.tokenize_path_line, workers
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
