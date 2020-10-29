# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from ncc import LOGGER
from ncc.utils.util_file import load_yaml
from ncc.tasks.codebert.masked_code_roberta import load_code_dataset_mlm
from ncc import tasks
import torch
from ncc.data import iterators
from ncc.logging import progress_bar
from ncc.utils import distributed_utils
from ncc.trainer.ncc_trainer import Trainer
import numpy as np
import random


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--config", "-c", default='javascript_mlm', type=str, help="load {language}.yml for train",
    )
    args = parser.parse_args()
    yaml_file = os.path.join(os.path.dirname(__file__), 'config', '{}.yml'.format(args.config))
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)

    torch.manual_seed(args['common']['seed'])
    torch.cuda.manual_seed(args['common']['seed'])
    np.random.seed(args['common']['seed'])
    random.seed(args['common']['seed'])

    task = tasks.setup_task(args)  # task.tokenizer
    model = task.build_model(args)  # , config
    criterion = task.build_criterion(args)
    LOGGER.info(model)
    LOGGER.info('model {}, criterion {}'.format(args['model']['arch'], criterion.__class__.__name__))
    LOGGER.info('num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    data_path = args['task']['data']
    split = 'train'
    combine = False
    src_dict = task.source_dictionary

    # Build trainer
    trainer = Trainer(args, task, model, criterion)

    epoch = 1
    # Option1: for hybrid debug
    # dataset = load_augmented_code_dataset_hybrid(args, epoch, data_path, split, src_dict, combine)
    # Option2: for coco debug
    # dataset = load_augmented_code_dataset_moco(args, epoch, data_path, split, src_dict, combine)
    # Option3: for mlm debug
    dataset = load_code_dataset_mlm(args, epoch, data_path, split, src_dict, combine)
    # item = dataset.__getitem__(0)
    # print('item: ')
    # print(item)
    # exit()

    epoch_itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=args['dataset']['max_tokens'],
        max_sentences=args['dataset']['max_sentences'],
        max_positions=args['task']['max_source_positions'],  # args['task']['max_source_positions'],
        ignore_invalid_inputs=True,
        required_batch_size_multiple=args['dataset']['required_batch_size_multiple'],
        seed=args['common']['seed'],
        num_shards=1,
        shard_id=0,
        num_workers=args['dataset']['num_workers'],
        epoch=1,
    )

    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args['distributed_training']['fix_batches_to_gpus'],
        shuffle=(epoch_itr.next_epoch_idx > args['dataset']['curriculum']),
    )
    # print('itr: ', itr)
    # for i, obj in enumerate(itr):
    #     print('i: ', i)
    #     print('obj: ', obj)
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

    # log_output = trainer.train_step([batch])
    # print('log_output: ', log_output)
    for samples in progress:
        # print('samples: ', samples)
        # exit()
        log_output = trainer.train_step(samples)
        print('log_output: ', log_output)
        # exit()
