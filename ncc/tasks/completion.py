# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from ncc import LOGGER
import torch
import numpy as np
from ncc.logging import metrics
from ncc.tasks.fairseq_task import FairseqTask
from ncc.tasks import register_task
from ncc.utils import utils
from ncc.data.tools import data_utils
from ncc.data.completion.trav_transformer_dataset import TravTransformerDataset
from ncc.data.completion.seqrnn_dataset import SeqRNNDataset
from ncc.data.dictionary import Dictionary
from ncc.utils import tokenizer
import json


def load_path_dataset(data_path, split, src, src_dict, dataset_impl):
    # def split_exists(split, src, tgt, lang, data_path):
    #     filename = os.path.join(data_path, '{}.{}'.format(split, src))  # -{}.{} , tgt, lang
    #     return indexed_dataset.dataset_exists(filename, impl=dataset_impl)
    #
    # # infer langcode
    # if split_exists(split, src, tgt, src, data_path):
    #     prefix = os.path.join(data_path, '{}.'.format(split_k))  # {}-{}. , src, tgt
    # elif split_exists(split, tgt, src, src, data_path):
    #     prefix = os.path.join(data_path, '{}.'.format(split_k))  # {}-{}. , tgt, src

    # prefix = os.path.join(data_path, '{}.'.format(split))

    # fp = os.path.join(prefix, 'train.ast_trav_df')
    # ids_fp = os.path.join(data_path, 'generated_ids.txt')
    # datasetup = Setup(data_path, fp=prefix+'ast_trav_df', ids_fp=prefix+'generated_ids.ast_trav_df', max_vocab=100000,
    #                   mode="train")
    # dataset = datasetup._create_dataset()

    source_path = os.path.join(data_path, '{}.ast_trav_df'.format(split))
    src_dataset = data_utils.load_indexed_dataset(source_path, 'ast', src_dict, dataset_impl)

    node_id_path = os.path.join(data_path, '{}.generate_id.ast_trav_df'.format(split))
    # node_ids = data_utils.load_indexed_dataset(node_id_path, 'node_id', src_dict, dataset_impl)
    node_ids = []
    with open(node_id_path, 'r', encoding='utf-8') as f:
        for ids_line in f:
            node_ids.append(json.loads(ids_line))

    return TravTransformerDataset(
        src_dataset, src_dataset.sizes, src_dict, node_ids,
    )


def load_tok_dataset(data_path, split, tgt_dict, dataset_impl, max_target_positions):
    target_path = os.path.join(data_path, '{}.tok'.format(split))
    tgt_dataset = data_utils.load_indexed_dataset(target_path, 'dfs', tgt_dict, dataset_impl)

    node_id_path = os.path.join(data_path, '{}.ids'.format(split))
    with open(node_id_path, 'r', encoding='utf-8') as f:
        node_ids = [json.loads(ids_line) for ids_line in f]

    return SeqRNNDataset(
        tgt_dataset, tgt_dataset.sizes, tgt_dict, node_ids, tgt_dataset.extends,
        max_target_positions=max_target_positions,
    )


def load_dfs_dataset(data_path, split, src, src_dict, dataset_impl):
    source_path = os.path.join(data_path, '{}.ast_trav_df'.format(split))
    src_dataset = data_utils.load_indexed_dataset(source_path, 'dfs', src_dict, dataset_impl)

    node_id_path = os.path.join(data_path, '{}.ids'.format(split))
    node_ids = []
    with open(node_id_path, 'r', encoding='utf-8') as f:
        for ids_line in f:
            node_ids.append(json.loads(ids_line))

    return SeqRNNDataset(
        src_dataset, src_dataset.sizes, src_dict, node_ids,
    )


@register_task('completion')
class CompletionTask(FairseqTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(args['task']['data'])
        assert len(paths) > 0
        # load dictionaries
        dictionary = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args['task']['source_lang'])))
        LOGGER.info('[{}] dictionary: {} types'.format(args['task']['source_lang'], len(dictionary)))
        return cls(args, dictionary)

    @classmethod
    def build_dictionary(
            cls, filenames, tokenize_func=tokenizer.tokenize_list,
            workers=1, threshold=-1, nwords=-1, padding_factor=8
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
        d = Dictionary()

        for filename in filenames:
            Dictionary.add_token_to_dictionary(
                filename, d, tokenize_func, workers
            )

        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args['task']['data'])
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        if self.args['model']['arch'] == 'trav_trans':
            self.datasets[split] = load_path_dataset(data_path, split, self.target_dictionary,
                                                     dataset_impl=self.args['dataset']['dataset_impl'])
        elif self.args['model']['arch'] == 'seqrnn':
            self.datasets[split] = load_tok_dataset(data_path, split, self.target_dictionary,
                                                    dataset_impl=self.args['dataset']['dataset_impl'],
                                                    max_target_positions=self.max_positions())
        elif self.args['model']['arch'] == 'traverse_transformer':
            self.datasets[split] = load_dfs_dataset(data_path, split, self.target_dictionary,
                                                    dataset_impl=self.args['dataset']['dataset_impl'])

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return SeqRNNDataset(src_tokens, src_lengths, self.target_dictionary)  # TODO: bug

    def build_model(self, args):
        model = super().build_model(args)

        if args['task']['eval_accuracy'] or args['task']['eval_mrr']:
            self.sequence_completor = self.build_completor([model], args)

        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        with torch.no_grad():
            net_output = self.sequence_completor.complete([model], sample, prefix_tokens=None)

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        loss_mask = sample['loss_mask']
        lprobs = lprobs.view(-1, lprobs.size(-1))
        lprobs = lprobs[loss_mask].contiguous()
        target = model.get_targets(sample, net_output).view(-1)
        target = target[loss_mask].contiguous()

        rank = torch.argmax(lprobs, 1)
        mrr = np.mean([1. / (r.item() + 1) for r in rank.view(-1)])

        ncorrect = torch.sum(rank == target)

        accuracy = ncorrect / sample['ntokens']
        logging_output['accuracy'] = accuracy
        logging_output['mrr'] = mrr

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            return sum(log.get(key, 0) for log in logging_outputs)

        if self.args['task']['eval_accuracy']:
            if sum_logs('accuracy') > 0:  # ==0: no accuracy items in the logging outputs, it means the training stage
                metrics.log_scalar('accuracy', sum_logs('accuracy'))
        if self.args['task']['eval_mrr']:
            if sum_logs('mrr') > 0:
                metrics.log_scalar('mrr', sum_logs('mrr'))

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return self.args['task']['max_target_positions']

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.dictionary
