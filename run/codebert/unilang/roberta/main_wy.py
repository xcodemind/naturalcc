#!/usr/bin/env python3 -u
# Copyright (c) NaturalCC, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import sys
sys.path.append('/data/wanyao/Dropbox/ghproj-titan/naturalcodev3')
import logging
import math
import os
import random
import numpy as np
import torch
from ncc.utils import checkpoint_utils, distributed_utils, utils
from ncc import tasks #, utils #  options,
from ncc.data import iterators
from ncc.log import meters, metrics, progress_bar
from ncc.trainer import Trainer
import argparse
from typing import Callable, List, Optional
from ncc.data.indexed_dataset import get_available_dataset_impl
from run.util import * #get_args
from ncc.utils.util_file import load_yaml

def single_main(config, init_distributed=False):
    # utils.import_user_module(config) # TODO: delete

    # assert config['dataset']['max_tokens'] is not None or config['dataset']['max_sentences'] is not None, \
    #     'Must specify batch size either with --max-tokens or --max-sentences'

    # Setup CUDA, GPU & distributed training
    if torch.cuda.is_available() and not config['common']['cpu']:
        torch.cuda.set_device(config['distributed_training']['device_id'])
    np.random.seed(config['common']['seed'])
    torch.manual_seed(config['common']['seed'])
    if init_distributed:
        config['distributed_training']['distributed_rank'] = distributed_utils.distributed_init(config)

    # Verify checkpoint directory
    if distributed_utils.is_master(config):
        checkpoint_utils.verify_checkpoint_directory(config['checkpoint']['save_dir'])

    # Print args
    LOGGER.info(config)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(config) # task.tokenizer

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in config['dataset']['valid_subset'].split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # Build model and criterion
    model = task.build_model(config)
    # model_config = task.build_model_config()

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
        if not config['dataset']['disable_validation'] and epoch_itr.epoch % config['dataset']['validate_interval'] == 0:
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
            LOGGER.info('early stop since valid performance hasn\'t improved for last {} runs'.format(config['checkpoint']['patience']))
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

    if config['model']['model_type'] in ['bert', 'roberta', 'distilbert', 'camembert'] and not config['task']['mlm']:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )
    # if config.eval_data_file is None and config.do_eval:
    #     raise ValueError(
    #         "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
    #         "or remove the --do_eval argument."
    #     )
    # if config.should_continue:
    #     sorted_checkpoints = _sorted_checkpoints(config)
    #     if len(sorted_checkpoints) == 0:
    #         raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
    #     else:
    #         config.model_name_or_path = sorted_checkpoints[-1]

    # if (
    #         os.path.exists(config.output_dir)
    #         and os.listdir(config.output_dir)
    #         and config.do_train
    #         and not config.overwrite_output_dir
    # ):
    #     raise ValueError(
    #         "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
    #             config.output_dir
    #         )
    #     )

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
            config=(config, ),
            nprocs=config['distributed_training']['distributed_world_size'],
        )
    else:
        # single GPU training
        print('single GPU training...')
        single_main(config)


if __name__ == '__main__':
    cli_main()