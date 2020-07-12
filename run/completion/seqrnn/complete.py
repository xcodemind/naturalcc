#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate the perplexity of a trained language model.
"""

import logging
import os
import math
import numpy as np
from collections import namedtuple
import torch
from ncc import LOGGER
from ncc import tasks
from ncc.utils import checkpoint_utils, distributed_utils
from ncc.logging import metrics, progress_bar
from ncc.utils import utils
from ncc.utils.util_file import load_yaml
from ncc.data.completion.lm_context_window_dataset import LMContextWindowDataset
from ncc.logging.meters import StopwatchMeter, TimeMeter
from ncc.eval.completion_scorer import CompletionScorer
import sys

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger('fairseq_cli.eval_lm')


class WordStat(object):
    def __init__(self, word, is_bpe):
        self.word = word
        self.is_bpe = is_bpe
        self.log_prob = 0
        self.next_word_prob = 0
        self.count = 0
        self.missing_next_words = 0

    def add(self, log_prob, next_word_prob):
        """ increments counters for the sum of log probs of current word and next
            word (given context ending at current word). Since the next word might be at the end of the example,
            or it might be not counted because it is not an ending subword unit,
            also keeps track of how many of those we have seen """
        if next_word_prob is not None:
            self.next_word_prob += next_word_prob
        else:
            self.missing_next_words += 1
        self.log_prob += log_prob
        self.count += 1

    def __str__(self):
        return '{}\t{}\t{}\t{}\t{}\t{}'.format(self.word, self.count, self.log_prob, self.is_bpe,
                                               self.next_word_prob, self.count - self.missing_next_words)


def main(parsed_args, **unused_kwargs):
    assert parsed_args['eval']['path'] is not None, '--path required for evaluation!'

    if torch.cuda.is_available() and not parsed_args['common']['cpu']:
        torch.cuda.set_device(parsed_args['distributed']['device_id'])

    utils.import_user_module(parsed_args)

    logger.info(parsed_args)

    use_cuda = torch.cuda.is_available() and not parsed_args['common']['cpu']

    task = tasks.setup_task(parsed_args)

    # Load ensemble
    logger.info('loading model(s) from {}'.format(parsed_args['eval']['path']))
    models, args = checkpoint_utils.load_model_ensemble(
        parsed_args['eval']['path'].split(os.pathsep),
        arg_overrides=eval(parsed_args['eval']['model_overrides']),
        task=task,
        suffix=parsed_args['eval']['checkpoint_suffix'],
    )

    task = tasks.setup_task(args)

    # Load dataset splits
    task.load_dataset(args['dataset']['gen_subset'])
    dataset = task.dataset(args['dataset']['gen_subset'])

    # Optimize ensemble for generation and set the source and dest dicts on the model (required by scorer)
    for model in models:
        model.make_generation_fast_()
        if args['common']['fp16']:
            model.half()
        if use_cuda:
            model.cuda()

    assert len(models) > 0

    logger.info('num. model params: {}'.format(sum(p.numel() for p in models[0].parameters())))

    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=args['dataset']['max_tokens'] or 36000,
        max_sentences=args['dataset']['max_sentences'],
        max_positions=utils.resolve_max_positions(*[
            model.max_positions() for model in models
        ]),
        ignore_invalid_inputs=True,
        num_shards=args['dataset']['num_shards'],
        shard_id=args['dataset']['shard_id'],
        num_workers=args['dataset']['num_workers'],
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args['common']['log_format'],
        log_interval=args['common']['log_interval'],
        default_log_format=('tqdm' if not args['common']['no_progress_bar'] else 'none'),
    )

    gen_timer = StopwatchMeter()
    scorer = CompletionScorer(task.target_dictionary)

    count = 0
    accuracy = 0.
    mrr = 0.

    for sample in progress:
        if 'net_input' not in sample:
            continue

        sample = utils.move_to_cuda(sample) if use_cuda else sample

        gen_timer.start()

        hypos = scorer.generate(models, sample)
        gen_timer.stop(sample['ntokens'])

        for i, hypos_i in enumerate(hypos):
            hypo = hypos_i[0]
            sample_id = sample['id'][i]
            tokens = hypo['tokens']
            tgt_len = tokens.numel()
            accuracy += hypo['accuracy']
            mrr += hypo['mrr']
            count += 1

        if count  > 100: # TODO: for debug
            break

        progress.log({'accuracy': accuracy/count, 'mrr': mrr/count})

    logger.info('Accuracy: {:.4f}, MRR: {:.4f}'.format(accuracy/count, mrr/count))


def cli_main():
    Argues = namedtuple('Argues', 'yaml')
    args_ = Argues('ruby_complete.yml')  # train_sl
    LOGGER.info(args_)
    # print(type(args.multi_processing))
    # assert False
    print('args: ', type(args_))
    # config = run_init(args.yaml, config=None)
    yaml_file = os.path.join(os.path.dirname(__file__), 'config', args_.yaml)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)

if __name__ == '__main__':
    cli_main()
