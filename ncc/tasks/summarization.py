# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import numpy as np
from argparse import Namespace
from ncc.logging import metrics
import itertools
from ncc import LOGGER
from ncc.data.dictionary import Dictionary
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
from ncc.data.summarization.language_pair_dataset import LanguagePairDataset
from ncc.data.summarization.path_dataset import PathDataset
from ncc.data.summarization.bin_ast_dataset import BinaryASTDataset
from ncc.utils.tokenizer import tokenize_string
from ncc.eval import eval_utils
EVAL_BLEU_ORDER = 4


def load_langpair_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions,
    max_target_positions, prepend_bos=False, load_alignments=False,
    truncate_source=False, append_source_id=False,
    append_eos_to_target=False,
):
    def split_exists(split, src, data_path):
        filename = os.path.join(data_path, '{}.{}'.format(split, src))  # -{}.{} , tgt, lang
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, data_path):
            prefix = os.path.join(data_path, '{}.'.format(split_k))  # {}-{}. , src, tgt
        elif split_exists(split_k, tgt, data_path):
            prefix = os.path.join(data_path, '{}.'.format(split_k))  # {}-{}. , tgt, src

        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

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

    # align_dataset = None
    # if load_alignments:
    #     align_path = os.path.join(data_path, '{}.align.{}-{}'.format(split, src, tgt))
    #     if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
    #         align_dataset = data_utils.load_indexed_dataset(align_path, None, dataset_impl)

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset_sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        align_dataset=None, eos=eos,
        remove_eos_from_source=True,
        append_eos_to_target=append_eos_to_target,
        shuffle=True,  # TODO debug: shuffle=False
    )


@register_task('summarization')
class SummarizationTask(FairseqTask):
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        paths = utils.split_paths(args['task']['data'])
        assert len(paths) > 0
        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], '{}.dict.json'.format(args['task']['source_lang'])))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], '{}.dict.json'.format(args['task']['target_lang'])))
        # src_dict = cls.load_dictionary(os.path.join(paths[0], '{}.dict.txt'.format(args['task']['source_lang'])))
        # tgt_dict = cls.load_dictionary(os.path.join(paths[0], '{}.dict.txt'.format(args['task']['target_lang'])))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        LOGGER.info('[{}] dictionary: {} types'.format(args['task']['source_lang'], len(src_dict)))
        LOGGER.info('[{}] dictionary: {} types'.format(args['task']['target_lang'], len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    @classmethod
    def build_dictionary(
        cls, filenames, tokenize_func=tokenize_string,
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
            append_eos_to_target=self.args['task']['append_eos_to_target'],
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def build_model(self, args):
        model = super().build_model(args)
        if args['task']['eval_bleu'] or args['task']['eval_rouge']:
            assert args['task']['eval_bleu_detok'] is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            # detok_args = json.loads(getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
            detok_args = json.loads(
                args['task']['eval_bleu_detok_args'] if args['task']['eval_bleu_detok_args'] else '{}')
            self.tokenizer = encoders.build_tokenizer(
                dict(
                    tokenizer=args['task']['eval_bleu_detok'] if args['task']['eval_bleu_detok'] else None,
                    # getattr(args, 'eval_bleu_detok', None),
                    **detok_args
                ))
            # The gen_args parameters have been set in the yml file
            # gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
            # self.sequence_generator = self.build_generator(Namespace(**gen_args))
            self.sequence_generator = self.build_generator(args)

        return model

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

    # def valid_step_(self, sample, model, criterion):
    #     loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
    #     if self.args['task']['eval_bleu']:
    #         bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
    #         print('bleu(valid_step): ', bleu)
    #         logging_output['_bleu_sys_len'] = bleu.sys_len
    #         logging_output['_bleu_ref_len'] = bleu.ref_len
    #         # we split counts into separate entries so that they can be
    #         # summed efficiently across workers using fast-stat-sync
    #         assert len(bleu.counts) == EVAL_BLEU_ORDER
    #         for i in range(EVAL_BLEU_ORDER):
    #             logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
    #             logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]
    #     # if self.args['task']['eval_rouge']:
    #     #     logging_output['_rouge'] = self._inference_with_rouge(self.rouge_sequence_generator, sample, model)
    #     return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        def decode(toks, escape_unk=False, trunc_eos=True):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args['task']['eval_bleu_remove_bpe'],
                escape_unk=escape_unk,
                trunc_eos=trunc_eos,
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        # gen_out = self.inference_step(generator, [model], sample, None)
        gen_out = self.sequence_generator.generate(model, sample)
        ids = sample['id'].tolist()
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]['tokens']))
            refs.append(decode(
                utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            ))

        bleu, rouge_l, meteor = self._inference_score(hyps, refs, ids)
        logging_output['bleu'] = bleu
        logging_output['rouge_l'] = rouge_l
        logging_output['meteor'] = meteor

        return loss, sample_size, logging_output
        # if self.args['task']['eval_bleu_print_samples']:
        #     LOGGER.info('example hypothesis: ' + hyps[0])
        #     LOGGER.info('example reference: ' + refs[0])
        # if self.args['task']['eval_bleu']:
        #     bleu = self._inference_with_bleu_wy(self.sequence_generator, sample, model)
            # print('bleu(valid_step): ', bleu)
            # gen_out = self.generator.generate(model, sample)
            # hyps, refs = [], []
            # for i in range(len(gen_out)):
            #     hyps.append(decode(gen_out[i]))
            #     refs.append(decode(
            #         utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
            #         escape_unk=True,  # don't count <unk> as matches to the hypo
            #     ))
            # bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            # logging_output['_bleu_sys_len'] = bleu.sys_len
            # logging_output['_bleu_ref_len'] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            # assert len(bleu.counts) == EVAL_BLEU_ORDER
            # for i in range(EVAL_BLEU_ORDER):
            #     logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
            #     logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]
            # print('bleu: ', bleu)
        # if self.args['task']['eval_rouge']:
        #     logging_output['_rouge'] = self._inference_with_rouge(self.rouge_sequence_generator, sample, model)
        #     logging_output['bleu'] = bleu
        # return bleu #loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args['task']['eval_bleu']:
            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            metrics.log_scalar('bleu', sum_logs('bleu'))

            # counts, totals = [], []
            # for i in range(EVAL_BLEU_ORDER):
            #     counts.append(sum_logs('_bleu_counts_' + str(i)))
            #     totals.append(sum_logs('_bleu_totals_' + str(i)))
            #
            # if max(totals) > 0:
            #     # log counts as numpy arrays -- log_scalar will sum them correctly
            #     metrics.log_scalar('_bleu_counts', np.array(counts))
            #     metrics.log_scalar('_bleu_totals', np.array(totals))
            #     metrics.log_scalar('_bleu_sys_len', sum_logs('_bleu_sys_len'))
            #     metrics.log_scalar('_bleu_ref_len', sum_logs('_bleu_ref_len'))
            #
            #     def compute_bleu(meters):
            #         import inspect
            #         import sacrebleu
            #         fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
            #         if 'smooth_method' in fn_sig:
            #             smooth = {'smooth_method': 'exp'}
            #         else:
            #             smooth = {'smooth': 'exp'}
            #         bleu = sacrebleu.compute_bleu(
            #             correct=meters['_bleu_counts'].sum,
            #             total=meters['_bleu_totals'].sum,
            #             sys_len=meters['_bleu_sys_len'].sum,
            #             ref_len=meters['_bleu_ref_len'].sum,
            #             **smooth
            #         )
            #         return round(bleu.score, 2)
            #
            #     metrics.log_derived('bleu', compute_bleu)





        # if self.args['task']['eval_rouge']:
        #
        #     if '_rouge' in logging_outputs[0]:
        #         metrics.log_scalar('_rouge_f',
        #                            sum(log['_rouge'][self.args['task']['eval_rouge_type']]['f'] \
        #                                for log in logging_outputs) / len(logging_outputs)
        #                            )
        #
        #         def compute_rouge(meters):
        #             if '_rouge_f' in meters:
        #                 return round(meters['_rouge_f'].avg, 2)
        #
        #         metrics.log_derived(self.args['task']['eval_rouge_type'], compute_rouge)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args['task']['max_source_positions'], self.args['task']['max_target_positions'])

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def _inference_score(self, hyps, refs, ids):
        hypotheses, references = dict(), dict()

        for key, pred, tgt in zip(ids, hyps, refs):
            hypotheses[key] = [pred]
            references[key] = tgt if isinstance(tgt, list) else [tgt]

        bleu, rouge_l, meteor = eval_utils.eval_accuracies(hypotheses, references, filename='pred.txt')

        return bleu, rouge_l, meteor

    # def _inference_with_bleu_wy(self, generator, sample, model):
    #     import sacrebleu
    #
    #     def decode(toks, escape_unk=False, trunc_eos=True):
    #         s = self.tgt_dict.string(
    #             toks.int().cpu(),
    #             self.args['task']['eval_bleu_remove_bpe'],
    #             escape_unk=escape_unk,
    #             trunc_eos=trunc_eos,
    #         )
    #         if self.tokenizer:
    #             s = self.tokenizer.decode(s)
    #         return s
    #
    #     # gen_out = self.inference_step(generator, [model], sample, None)
    #     gen_out = generator.generate(model, sample)
    #     hyps, refs = [], []
    #     for i in range(len(gen_out)):
    #         # hypo = gen_out[i][0]['tokens']
    #         # hyps.append(decode(hypo[:hypo.index(self.tgt_dict.eos())]))
    #         hyps.append(decode(gen_out[i][0]['tokens']))
    #
    #         refs.append(decode(
    #             utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
    #             escape_unk=True,  # don't count <unk> as matches to the hypo
    #         ))
    #     if self.args['task']['eval_bleu_print_samples']:
    #         LOGGER.info('example hypothesis: ' + hyps[0])
    #         LOGGER.info('example reference: ' + refs[0])
    #     # tokenize = sacrebleu.DEFAULT_TOKENIZER if not self.args['task']['eval_tokenized_bleu'] else 'none'
    #     # return sacrebleu.corpus_bleu(hyps, [refs], tokenize=tokenize)
    #     # if self.args['task']['eval_tokenized_bleu']:
    #     #     bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize='none')
    #     # else:
    #     #     bleu = sacrebleu.corpus_bleu(hyps, [refs])
    #
    #
    #     return bleu
    #
    # def _inference_with_bleu(self, generator, sample, model):
    #     import sacrebleu
    #
    #     def decode(toks, escape_unk=False):
    #         s = self.tgt_dict.string(
    #             toks.int().cpu(),
    #             self.args['task']['eval_bleu_remove_bpe'],
    #             escape_unk=escape_unk,
    #         )
    #         if self.tokenizer:
    #             s = self.tokenizer.decode(s)
    #         return s
    #
    #     # gen_out = self.inference_step(generator, [model], sample, None)
    #     gen_out = generator.generate([model], sample)
    #     hyps, refs = [], []
    #     for i in range(len(gen_out)):
    #         hyps.append(decode(gen_out[i][0]['tokens']))
    #         refs.append(decode(
    #             utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
    #             escape_unk=True,  # don't count <unk> as matches to the hypo
    #         ))
    #     if self.args['task']['eval_bleu_print_samples']:
    #         LOGGER.info('example hypothesis: ' + hyps[0])
    #         LOGGER.info('example reference: ' + refs[0])
    #     # tokenize = sacrebleu.DEFAULT_TOKENIZER if not self.args['task']['eval_tokenized_bleu'] else 'none'
    #     # return sacrebleu.corpus_bleu(hyps, [refs], tokenize=tokenize)
    #     if self.args['task']['eval_tokenized_bleu']:
    #         return sacrebleu.corpus_bleu(hyps, [refs], tokenize='none')
    #     else:
    #         return sacrebleu.corpus_bleu(hyps, [refs])

    # def _inference_with_rouge(self, generator, sample, model):
    #     from rouge import Rouge
    #
    #     def decode(toks, escape_unk=False):
    #         s = self.tgt_dict.string(
    #             toks.int().cpu(),
    #             self.args['task']['eval_rouge_remove_bpe'],
    #             escape_unk=escape_unk,
    #         )
    #         if self.tokenizer:
    #             s = self.tokenizer.decode(s)
    #         return s
    #
    #     gen_out = self.inference_step(generator, [model], sample, None)
    #     hyps, refs = [], []
    #     for i in range(len(gen_out)):
    #         hyps.append(decode(gen_out[i][0]['tokens']))
    #         refs.append(decode(
    #             utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
    #             escape_unk=True,  # don't count <unk> as matches to the hypo
    #         ))
    #     if self.args['task']['eval_rouge_print_samples']:
    #         LOGGER.info('example hypothesis: ' + hyps[0])
    #         LOGGER.info('example reference: ' + refs[0])
    #     # tokenize = sacrebleu.DEFAULT_TOKENIZER if not self.args['task']['eval_tokenized_bleu'] else 'none'
    #     # return sacrebleu.corpus_bleu(hyps, [refs], tokenize=tokenize)
    #     rouge_scores = Rouge().get_scores(hyps, refs, avg=True)
    #     return rouge_scores
