# -*- coding: utf-8 -*-

# !/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate the perplexity of a trained language model.
"""
import os
import numpy as np
from collections import namedtuple
import random
import torch
import torch.nn.functional as F
from ncc import LOGGER
from ncc import tasks
from ncc.utils import checkpoint_utils
from ncc.logging import metrics, progress_bar
from ncc.utils import utils
from ncc.utils.util_file import load_yaml
from ncc.logging.meters import StopwatchMeter
from ncc.eval.com2cod_retrieval import Com2CodeRetrievalScorer


def main(args, **unused_kwargs):
    assert args['eval']['path'] is not None, '--path required for evaluation!'

    if torch.cuda.is_available() and not args['common']['cpu']:
        torch.cuda.set_device(args['distributed_training']['device_id'])

    LOGGER.info(args)
    use_cuda = torch.cuda.is_available() and not args['common']['cpu']
    task = tasks.setup_task(args)

    # Load ensemble
    LOGGER.info('loading model(s) from {}'.format(args['eval']['path']))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args['eval']['path']),
        arg_overrides=eval(args['eval']['model_overrides']),
        task=task,
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

    LOGGER.info('num. model params: {}'.format(sum(p.numel() for p in models[0].parameters())))

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

    code_reprs, query_reprs = [], []
    for sample in progress:
        if 'net_input' not in sample:
            continue
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        batch_code_reprs, batch_query_reprs = models[0](**sample['net_input'])
        code_reprs.extend(batch_code_reprs.tolist())
        query_reprs.extend(batch_query_reprs.tolist())
    code_reprs = np.asarray(code_reprs, dtype=np.float32)
    query_reprs = np.asarray(query_reprs, dtype=np.float32)

    assert code_reprs.shape == query_reprs.shape, (code_reprs.shape, query_reprs.shape)
    eval_size = len(code_reprs) if args['eval']['eval_size'] == -1 else args['eval']['eval_size']

    MRR = []
    for idx in range(len(query_reprs)):
        if eval_size == -1:
            batch_ids = range(len(query_reprs))
            gt_idx = idx
        else:
            batch_ids = set(random.sample(range(len(query_reprs)), eval_size))
            if idx not in batch_ids:
                batch_ids = list(batch_ids)[:eval_size - 1] + [idx]
                gt_idx = eval_size - 1
            else:
                batch_ids = list(batch_ids)
                gt_idx = batch_ids.index(idx)
        batch_code_reprs = torch.from_numpy(code_reprs[batch_ids, :])
        batch_query_reprs = torch.from_numpy(query_reprs[batch_ids, :])
        similarity_scores = F.cosine_similarity(batch_code_reprs, batch_query_reprs)
        gt_sim = similarity_scores[gt_idx]
        MRR.append(
            (1 / (similarity_scores >= gt_sim).sum(dim=-1).float()).item()
        )
    print('mrr: {:.4f}'.format(sum(MRR) / len(MRR)))


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--language", "-l", default='javascript', type=str, help="load {language}.yml for train",
    )
    args = parser.parse_args()
    yaml_file = os.path.join(os.path.dirname(__file__), 'config', '{}.yml'.format(args.language))
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == '__main__':
    cli_main()
