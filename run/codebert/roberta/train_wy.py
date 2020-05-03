#!/usr/bin/env python3 -u
# Copyright (c) NaturalCC, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""
import sys
import logging
import math
import os
import random
import numpy as np
import torch
from ncc.utils import checkpoint_utils, distributed_utils, utils
from ncc import tasks  # , utils #  options,
from ncc.data import iterators
from ncc.logging import meters, metrics, progress_bar
from ncc.trainer.fair_trainer import Trainer
import argparse
from typing import Callable, List, Optional
from ncc.data.indexed_dataset import get_available_dataset_impl
from run.util import *  # get_args
from ncc.utils.util_file import load_yaml


@metrics.aggregate('train')
def train(config, trainer, task, epoch_itr):
    """Train the model for one epoch."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=config['distributed_training']['fix_batches_to_gpus'],
        shuffle=(epoch_itr.next_epoch_idx > config['dataset']['curriculum']),
    )
    update_freq = (
        config['optimization']['update_freq'][epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(config['optimization']['update_freq'])
        else config['optimization']['update_freq'][-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.progress_bar(
        itr,
        log_format=config['common']['log_format'],
        log_interval=config['common']['log_interval'],
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            config['common']['tensorboard_logdir'] if distributed_utils.is_master(config) else None
        ),
        default_log_format=('tqdm' if not config['common']['no_progress_bar'] else 'simple'),
    )

    # task specific setup per epoch
    task.begin_epoch(epoch_itr.epoch, trainer.get_model())

    valid_subsets = config['dataset']['valid_subset'].split(',')
    max_update = config['optimization']['max_update'] or math.inf
    for samples in progress:
        with metrics.aggregate('train_inner'):
            log_output = trainer.train_step(samples)
            if log_output is None:  # OOM, overflow, ...
                continue

        # log mid-epoch stats
        num_updates = trainer.get_num_updates()
        if num_updates % config['common']['log_interval'] == 0:
            stats = get_training_stats(metrics.get_smoothed_values('train_inner'))
            progress.log(stats, tag='train_inner', step=num_updates)

            # reset mid-epoch stats after each log interval
            # the end-of-epoch stats will still be preserved
            metrics.reset_meters('train_inner')

        if (
                not config['dataset']['disable_validation']
                and config['checkpoint']['save_interval_updates'] > 0
                and num_updates % config['checkpoint']['save_interval_updates'] == 0
                and num_updates > 0
        ):
            valid_losses = validate(config, trainer, task, epoch_itr, valid_subsets)
            checkpoint_utils.save_checkpoint(config, trainer, epoch_itr, valid_losses[0])

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


def should_stop_early(config, valid_loss):
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if config['checkpoint']['patience'] <= 0:
        return False

    def is_better(a, b):
        return a > b if config['checkpoint']['maximize_best_checkpoint_metric'] else a < b

    prev_best = getattr(should_stop_early, 'best', None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        return should_stop_early.num_runs >= config['checkpoint']['patience']


def validate(config, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""

    if config['dataset']['fixed_validation_seed'] is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(config['dataset']['fixed_validation_seed'])

    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=config['dataset']['max_tokens_valid'],
            max_sentences=config['dataset']['max_sentences_valid'],
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=config['dataset']['skip_invalid_size_inputs_valid_test'],
            required_batch_size_multiple=config['dataset']['required_batch_size_multiple'],
            seed=config['common']['seed'],
            num_shards=config['distributed_training']['distributed_world_size'],
            shard_id=config['distributed_training']['distributed_rank'],
            num_workers=config['dataset']['num_workers'],
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=config['common']['log_format'],
            log_interval=config['common']['log_interval'],
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                config['common']['tensorboard_logdir'] if distributed_utils.is_master(config) else None
            ),
            default_log_format=('tqdm' if not config['common']['no_progress_bar'] else 'simple'),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for sample in progress:
                trainer.valid_step(sample)

        # log validation stats
        stats = get_valid_stats(config, trainer, agg.get_smoothed_values())
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[config['checkpoint']['best_checkpoint_metric']])
    return valid_losses


def get_valid_stats(config, trainer, stats):
    if 'nll_loss' in stats and 'ppl' not in stats:
        stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        key = 'best_{0}'.format(config['checkpoint']['best_checkpoint_metric'])
        best_function = max if config['checkpoint']['maximize_best_checkpoint_metric'] else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[config['checkpoint']['best_checkpoint_metric']],
        )
    return stats


def single_main(config, init_distributed=False):
    # utils.import_user_module(config) # TODO: delete

    assert config['dataset']['max_tokens'] is not None or config['dataset']['max_sentences'] is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not config['common']['cpu']:
        torch.cuda.set_device(config['distributed_training']['device_id'])
    np.random.seed(config['common']['seed'])
    torch.manual_seed(config['common']['seed'])
    if init_distributed:
        config['distributed_training']['distributed_rank'] = distributed_utils.distributed_init(config)

    if distributed_utils.is_master(config):
        checkpoint_utils.verify_checkpoint_directory(config['checkpoint']['save_dir'])

    # Print args
    LOGGER.info(config)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(config)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in config['dataset']['valid_subset'].split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # Build model and criterion
    model = task.build_model(config)
    criterion = task.build_criterion(config)
    LOGGER.info(model)
    LOGGER.info('model {}, criterion {}'.format(config['model']['arch'], criterion.__class__.__name__))
    LOGGER.info('num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Build trainer
    trainer = Trainer(config, task, model, criterion)
    LOGGER.info('training on {} GPUs'.format(config['distributed_training']['distributed_world_size']))
    LOGGER.info('max tokens per GPU = {} and max sentences per GPU = {}'.format(
        config['dataset']['max_tokens'],
        config['dataset']['max_sentences'],
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(config, trainer)

    # Train until the learning rate gets too small
    max_epoch = config['optimization']['max_epoch'] or math.inf
    max_update = config['optimization']['max_update'] or math.inf
    lr = trainer.get_lr()
    train_meter = meters.StopwatchMeter()
    train_meter.start()
    valid_subsets = config['dataset']['valid_subset'].split(',')
    while (
            lr > config['optimization']['min_lr']
            and epoch_itr.next_epoch_idx <= max_epoch
            and trainer.get_num_updates() < max_update
    ):
        # train for one epoch
        train(config, trainer, task, epoch_itr)
        sys.exit()
        if not config['dataset']['disable_validation'] and epoch_itr.epoch % config['dataset'][
            'validate_interval'] == 0:
            valid_losses = validate(config, trainer, task, epoch_itr, valid_subsets)
        else:
            valid_losses = [None]

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint
        if epoch_itr.epoch % config['checkpoint']['save_interval'] == 0:
            checkpoint_utils.save_checkpoint(config, trainer, epoch_itr, valid_losses[0])

        # early stop
        if should_stop_early(config, valid_losses[0]):
            LOGGER.info('early stop since valid performance hasn\'t improved for last {} runs'.format(
                config['checkpoint']['patience']))
            break

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=(os.pathsep in getattr(config, 'data', '')),
        )
    train_meter.stop()
    LOGGER.info('done training in {:.1f} seconds'.format(train_meter.sum))


def distributed_main(i, config, start_rank=0):
    config['distributed_training']['device_id'] = i
    if config['distributed_training']['distributed_rank'] is None:  # torch.multiprocessing.spawn
        config['distributed_training']['distributed_rank'] = start_rank + i
    single_main(config, init_distributed=True)


def cli_main():
    # args = get_args()
    # dataset_dir = None, dataset_type = None, debug = 0, device = 0, lang_mode = None, log_root_dir = None, method_name = None, multi_processing = 0, occupy_gpu = 'no', save_dir = None, task = None, train_mode = None, yaml = 'wiki.yml'
    # Argues = namedtuple('Argues', 'yaml task lang_mode method_name train_mode dataset_type multi_processing')
    Argues = namedtuple('Argues', 'yaml')

    args = Argues('wiki.yml')  # train_sl
    LOGGER.info(args)
    # print(type(args.multi_processing))
    # assert False
    print('args: ', type(args))
    # config = run_init(args.yaml, config=None)
    yaml_file = os.path.join(sys.path[0], args.yaml)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    config = load_yaml(yaml_file)

    LOGGER.info(config)

    if config['distributed_training']['distributed_init_method'] is None:
        distributed_utils.infer_init_method(config)

    if config['distributed_training']['distributed_init_method'] is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not config['distributed_training']['distributed_no_spawn']:
            start_rank = config['distributed_training']['distributed_rank']
            config['distributed_training']['distributed_rank'] = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                config=(config, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(config['distributed_training']['device_id'], config)
    elif config['distributed_training']['distributed_world_size'] > 1:
        # fallback for single node with multiple GPUs
        assert config['distributed_training']['distributed_world_size'] <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        config['distributed_training']['distributed_init_method'] = 'tcp://localhost:{port}'.format(port=port)
        config['distributed_training']['distributed_rank'] = None  # set based on device id
        torch.multiprocessing.spawn(
            fn=distributed_main,
            config=(config,),
            nprocs=config['distributed_training']['distributed_world_size'],
        )
    else:
        # single GPU training
        print('single GPU training...')
        single_main(config)


if __name__ == '__main__':
    cli_main()
