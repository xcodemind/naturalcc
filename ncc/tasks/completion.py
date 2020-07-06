# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from ncc import LOGGER
from argparse import Namespace
import numpy as np
from ncc.logging import metrics
from ncc.tasks.fairseq_task import FairseqTask
from ncc.tasks import register_task
from ncc.utils import utils
from ncc.data.tools import data_utils
from ncc.data.completion.trav_transformer_dataset import TravTransformerDataset
from ncc.data.completion.seqrnn_dataset import SeqRNNDataset
from ncc.data.dictionary import Dictionary
from ncc.utils import tokenizer  # , utils # metrics, search,
import json

EVAL_BLEU_ORDER = 4


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


def load_tok_dataset(data_path, split, src, src_dict, dataset_impl):
    source_path = os.path.join(data_path, '{}.tok'.format(split))
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
        # self.dictionary = dictionary
        # self.seed = args['common']['seed']
        self.args = args
        # # add mask token
        # self.mask_idx = dictionary.add_symbol('<mask>')、
        # self.tokenizer = tokenizer
        # self.mask_idx = self.tokenizer.mask_token_id
        self.dictionary = dictionary
        # self.tgt_dict = tgt_dict

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
        # tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args['task']['target_lang'])))
        # assert src_dict.pad() == tgt_dict.pad()
        # assert src_dict.eos() == tgt_dict.eos()
        # assert src_dict.unk() == tgt_dict.unk()
        # LOGGER.info('[{}] dictionary: {} types'.format(args['task']['source_lang'], len(dictionary)))
        # LOGGER.info('[{}] dictionary: {} types'.format(args['task']['target_lang'], len(tgt_dict)))

        return cls(args, dictionary)

    # overwrite the build_dictionary function
    @classmethod
    def build_dictionary(
            cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8
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
                filename, d, tokenizer.tokenize_list, workers
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

        # infer langcode
        src = self.args['task']['source_lang']

        if self.args['model']['arch'] == 'trav_trans':
            self.datasets[split] = load_path_dataset(data_path, split, src, self.source_dictionary,
                                                     dataset_impl=self.args['dataset']['dataset_impl'])
        elif self.args['model']['arch'] == 'seqrnn':
            self.datasets[split] = load_tok_dataset(data_path, split, src, self.source_dictionary,
                                                    dataset_impl=self.args['dataset']['dataset_impl'])
        # print('self.datasets[split]: ', self.datasets[split].__getitem__(0))

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return SeqRNNDataset(src_tokens, src_lengths, self.source_dictionary)  # TODO: bug

    def build_model(self, args):
        model = super().build_model(args)
        if getattr(args, 'eval_bleu', False):
            # assert getattr(args, 'eval_bleu_detok', None) is not None, (
            #     '--eval-bleu-detok is required if using --eval-bleu; '
            #     'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
            #     'to disable detokenization, e.g., when using sentencepiece)'
            # )
            # detok_args = json.loads(getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
            # self.tokenizer = encoders.build_tokenizer(Namespace(
            #     tokenizer=getattr(args, 'eval_bleu_detok', None),
            #     **detok_args
            # ))

            # gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
            self.sequence_completor = self.build_completor([model], Namespace(**gen_args))
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args['task']['eval_bleu']:
            bleu = self._inference_with_bleu(self.sequence_completor, sample, model)
            logging_output['_bleu_sys_len'] = bleu.sys_len
            logging_output['_bleu_ref_len'] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
                logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        if self.args['task']['eval_accuracy']:

            metrics.log_scalar('accuracy', 1.111)
        if self.args['task']['eval_mrr']:
            metrics.log_scalar('mrr', 2.222)
        if self.args['task']['eval_bleu']:

            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs('_bleu_counts_' + str(i)))
                totals.append(sum_logs('_bleu_totals_' + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar('_bleu_counts', np.array(counts))
                metrics.log_scalar('_bleu_totals', np.array(totals))
                metrics.log_scalar('_bleu_sys_len', sum_logs('_bleu_sys_len'))
                metrics.log_scalar('_bleu_ref_len', sum_logs('_bleu_ref_len'))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu
                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if 'smooth_method' in fn_sig:
                        smooth = {'smooth_method': 'exp'}
                    else:
                        smooth = {'smooth': 'exp'}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters['_bleu_counts'].sum,
                        total=meters['_bleu_totals'].sum,
                        sys_len=meters['_bleu_sys_len'].sum,
                        ref_len=meters['_bleu_ref_len'].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived('bleu', compute_bleu)


    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.dictionary
