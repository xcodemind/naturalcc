#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import os
import math
import random
import numpy as np
import json
import collections
from collections import namedtuple
import torch
from ncc import LOGGER
from ncc import tasks
from ncc.logging import meters
from ncc.trainer.fair_trainer import Trainer
from ncc.utils import checkpoint_utils, distributed_utils
from ncc.utils.util_file import load_yaml
from ncc.logging import metrics, progress_bar
from ncc.utils import utils
from ncc.utils.file_utils import remove_files
from ncc.data import iterators
from ncc.utils.fed_utils import save_expert_outputs
from ncc.eval import bleu_scorer
from ncc.eval.old_sequence_generator import SequenceGenerator


@metrics.aggregate('train')
def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args['distributed_training']['fix_batches_to_gpus'],
        shuffle=(epoch_itr.next_epoch_idx > args['dataset']['curriculum']),
    )
    update_freq = (
        args['optimization']['update_freq'][epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(args['optimization']['update_freq'])
        else args['optimization']['update_freq'][-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args['common']['log_format'],
        log_interval=args['common']['log_interval'],
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            args['common']['tensorboard_logdir'] if distributed_utils.is_master(args) else None
        ),
        default_log_format=('tqdm' if not args['common']['no_progress_bar'] else 'simple'),
    )

    # task specific setup per epoch
    task.begin_epoch(epoch_itr.epoch, trainer.get_model())

    valid_subsets = args['dataset']['valid_subset'].split(',')
    max_update = args['optimization']['max_update'] or math.inf
    for samples in progress:
        with metrics.aggregate('train_inner'):
            log_output = trainer.train_step(samples)
            if log_output is None:  # OOM, overflow, ...
                continue

        # log mid-epoch stats
        num_updates = trainer.get_num_updates()
        if num_updates % args['common']['log_interval'] == 0:
            stats = get_training_stats(metrics.get_smoothed_values('train_inner'))
            progress.log(stats, tag='train_inner', step=num_updates)

            # reset epoch-level meters
            metrics.reset_meters('train_inner')

        if (
            not args['dataset']['disable_validation']
            and args['checkpoint']['save_interval_updates'] > 0
            and num_updates % args['checkpoint']['save_interval_updates'] == 0
            and num_updates > 0
        ):
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = get_training_stats(metrics.get_smoothed_values('train'))
    progress.print(stats, tag='train', step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters('train')


def get_training_stats(stats):
    if 'nll_loss' in stats and 'ppl' not in stats:
        stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
    stats['wall'] = round(metrics.get_meter('default', 'wall').elapsed_time, 0)
    return stats


def single_main(args, init_distributed=False):
    assert args['dataset']['max_tokens'] is not None or args['dataset']['max_sentences'] is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'
    metrics.reset()

    # 0. Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args['common']['cpu']:
        torch.cuda.set_device(args['distributed_training']['device_id'])
    np.random.seed(args['common']['seed'])
    torch.manual_seed(args['common']['seed'])
    if init_distributed:
        args['distributed_training']['distributed_rank'] = distributed_utils.distributed_init(args)

    # Verify checkpoint directory
    if distributed_utils.is_master(args):
        save_dir = args['checkpoint']['save_dir']
        checkpoint_utils.verify_checkpoint_directory(save_dir)
        remove_files(save_dir, 'pt')

    # Print args
    LOGGER.info(args)

    # 1. Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # 2. Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args['dataset']['valid_subset'].split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # 3. Build model and criterion
    model = task.build_model(args)
    # load pretrained model
    assert args['checkpoint']['pretrain_path']
    LOGGER.info('Load pretrain model parameters from {}'.format(args['checkpoint']['pretrain_path']))
    state = checkpoint_utils.load_checkpoint_to_cpu(args['checkpoint']['pretrain_path'])
    model.load_state_dict(state["model"], strict=True, args=args)
    del state
    criterion = task.build_criterion(args)
    LOGGER.info(model)
    LOGGER.info('model {}, criterion {}'.format(args['model']['arch'], criterion.__class__.__name__))
    LOGGER.info('num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    trainer = Trainer(args, task, model, criterion)
    LOGGER.info('training on {} GPUs'.format(args['distributed_training']['distributed_world_size']))
    LOGGER.info('max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args['dataset']['max_tokens'],
        args['dataset']['max_sentences'],
    ))

    # save best bleu score of valid model
    valid_subsets = args['dataset']['valid_subset'].split(',')
    for subset in valid_subsets:
        # Initialize data iterator
        def get_itr():
            itr = task.get_batch_iterator(
                dataset=task.dataset(subset),
                max_tokens=args['dataset']['max_tokens_valid'],
                max_sentences=args['dataset']['max_sentences_valid'],
                max_positions=utils.resolve_max_positions(
                    task.max_positions(),
                    trainer.get_model().max_positions(),
                ),
                ignore_invalid_inputs=args['dataset']['skip_invalid_size_inputs_valid_test'],
                required_batch_size_multiple=args['dataset']['required_batch_size_multiple'],
                seed=args['common']['seed'],
                num_shards=args['distributed_training']['distributed_world_size'],
                shard_id=args['distributed_training']['distributed_rank'],
                num_workers=args['dataset']['num_workers'],
            ).next_epoch_itr(shuffle=False)
            progress = progress_bar.progress_bar(
                itr,
                log_format=args['common']['log_format'],
                log_interval=args['common']['log_interval'],
                epoch=0,
                prefix=f"valid on '{subset}' subset",
                tensorboard_logdir=(
                    args['common']['tensorboard_logdir'] if distributed_utils.is_master(args) else None
                ),
                default_log_format=('tqdm' if not args['common']['no_progress_bar'] else 'simple'),
            )
            return progress

    bleu_dict = {}
    for ds_id in range(num_dataset):
        if sample_size[ds_id].item() > 0:
            name = "bleu_" + task.dataset(subset).dataset_names[ds_id]
            bleu_dict[name] = stats[name] = bleu_scores[ds_id].item() / sample_size[ds_id].item()
            try:
                train_ds_id = task.dataset('train').dataset_names.index(
                    task.dataset(subset).dataset_names[ds_id])
                task.dataset('train').student_scores[train_ds_id] = bleu_dict[name]
            except ValueError:
                pass
    output_path = os.path.join(args['checkpoint']['save_dir'], 'val_bleu.json')
    json.dump(bleu_dict, open(output_path, 'w'))

    # save topk indices/probabities of best bleu model
    if args['checkpoint']['save_output']:
        save_expert_outputs(args, task, trainer)


def distributed_main(i, args, start_rank=0):
    args['distributed_training']['device_id'] = i
    if args['distributed_training']['distributed_rank'] is None:  # torch.multiprocessing.spawn
        args['distributed_training']['distributed_rank'] = start_rank + i
    single_main(args, init_distributed=True)


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--language", "-l", default='ruby', type=str, help="load {language}.yml for train",
    )
    parser.add_argument(
        "--train-mode", "-m", default='teacher', type=str, choices=['teacher', 'student', 'finetune'],
        help="False for training a teacher network on different datasets and generate topk probabilities and indices on datasets" \
             "True for distill dark knowledge from some teacher networks(implemented by generated topk probabilities and indices)",
    )
    args = parser.parse_args()
    # Argues = namedtuple('Argues', 'yaml')
    # args_ = Argues('ruby.yml')
    yaml_file = os.path.join(os.path.dirname(__file__), 'config', args.train_mode, '{}.yml'.format(args.language))
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)

    if args['distributed_training']['distributed_init_method'] is None:
        distributed_utils.infer_init_method(args)

    if args['distributed_training']['distributed_init_method'] is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args['distributed_training']['distributed_no_spawn']:
            start_rank = args['distributed_training']['distributed_rank']
            args['distributed_training']['distributed_rank'] = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args['distributed_training']['device_id'], args)
    elif args['distributed_training']['distributed_world_size'] > 1:
        # fallback for single node with multiple GPUs
        assert args['distributed_training']['distributed_world_size'] <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args['distributed_training']['distributed_init_method'] = 'tcp://localhost:{port}'.format(port=port)
        args['distributed_training']['distributed_rank'] = None  # set based on device id
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args,),
            nprocs=args['distributed_training']['distributed_world_size'],
        )
    else:
        LOGGER.info('single GPU training...')
        single_main(args)


if __name__ == '__main__':
    cli_main()
