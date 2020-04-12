# -*- coding: utf-8 -*-
import os
import sys
import math
import random
import numpy as np
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
from ncc.data import iterators


def single_main(args, init_distributed=False):
    # utils.import_user_module(args) # TODO: delete

    # assert args['dataset']['max_tokens'] is not None or args['dataset']['max_sentences'] is not None, \
    #     'Must specify batch size either with --max-tokens or --max-sentences'

    # Setup CUDA, GPU & distributed training
    if torch.cuda.is_available() and not args['common']['cpu']:
        torch.cuda.set_device(args['distributed_training']['device_id'])
    np.random.seed(args['common']['seed'])
    torch.manual_seed(args['common']['seed'])
    if init_distributed:
        args['distributed_training']['distributed_rank'] = distributed_utils.distributed_init(args)

    # Verify checkpoint directory
    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args['checkpoint']['save_dir'])

    # Print args
    LOGGER.info(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args) # task.tokenizer
    # build model_config
    config = task.build_config(args)
    # Build model and criterion
    model = task.build_model(args, config)
    # model_config = task.build_model_config()

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    # for valid_sub_split in args['dataset']['valid_subset'].split(','):
    #     task.load_dataset(valid_sub_split, combine=False, epoch=1)

    criterion = task.build_criterion(args)
    LOGGER.info(model)
    LOGGER.info('model {}, criterion {}'.format(args['model']['arch'], criterion.__class__.__name__))
    LOGGER.info('num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    LOGGER.info('training on {} GPUs'.format(args['distributed_training']['distributed_world_size']))
    # LOGGER.info('max tokens per GPU = {} and max sentences per GPU = {}'.format(
    #     args['dataset']['max_tokens'],
    #     args['dataset']['max_sentences'],
    # ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer, combine=False)

    # Train until the learning rate gets too small
    max_epoch = args['optimization']['max_epoch'] or math.inf
    max_update = args['optimization']['max_update'] or math.inf
    lr = trainer.get_lr()
    train_meter = meters.StopwatchMeter()
    train_meter.start()
    valid_subsets = args['dataset']['valid_subset'].split(',')
    while (
        lr > args['optimization']['min_lr']
        and epoch_itr.next_epoch_idx <= max_epoch
        and trainer.get_num_updates() < max_update
    ):
        # train for one epoch
        valid_losses = train(args, trainer, task, epoch_itr) # max_update
        if not args['dataset']['disable_validation'] and epoch_itr.epoch % args['dataset']['validate_interval'] == 0:
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
        else:
            valid_losses = [None]

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint
        if epoch_itr.epoch % args['checkpoint']['save_interval'] == 0:
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        # early stop
        if should_stop_early(args, valid_losses[0]):
            LOGGER.info('early stop since valid performance hasn\'t improved for last {} runs'.format(args['checkpoint']['patience']))
            break

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            combine=False,
            # sharded data: get train iterator for next epoch
            load_dataset=(os.pathsep in getattr(args, 'data', '')),
        )
    train_meter.stop()
    LOGGER.info('done training in {:.1f} seconds'.format(train_meter.sum))


def distributed_main(i, args, start_rank=0):
    args['distributed_training']['device_id'] = i
    if args['distributed_training']['distributed_rank'] is None:  # torch.multiprocessing.spawn
        args['distributed_training']['distributed_rank'] = start_rank + i
    single_main(args, init_distributed=True)


def cli_main():
    # args = get_args()
    # dataset_dir = None, dataset_type = None, debug = 0, device = 0, lang_mode = None, log_root_dir = None, method_name = None, multi_processing = 0, occupy_gpu = 'no', save_dir = None, task = None, train_mode = None, yaml = 'wiki.yml'
    # Argues = namedtuple('Argues', 'yaml task lang_mode method_name train_mode dataset_type multi_processing')
    Argues = namedtuple('Argues', 'yaml')

    args_ = Argues('ruby.yml')  # train_sl
    LOGGER.info(args_)
    # print(type(args.multi_processing))
    # assert False
    print('args: ', type(args_))
    # config = run_init(args.yaml, config=None)
    yaml_file = os.path.join(sys.path[0], args_.yaml)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)

    LOGGER.info(args)

    # if args['model']['arch'] in ['bert', 'roberta', 'distilbert', 'camembert'] and not args['task']['mlm']:
    #     raise ValueError(
    #         "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
    #         "flag (masked language modeling)."
    #     )
    # if args.eval_data_file is None and args.do_eval:
    #     raise ValueError(
    #         "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
    #         "or remove the --do_eval argument."
    #     )
    # if args.should_continue:
    #     sorted_checkpoints = _sorted_checkpoints(args)
    #     if len(sorted_checkpoints) == 0:
    #         raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
    #     else:
    #         args.model_name_or_path = sorted_checkpoints[-1]

    # if (
    #         os.path.exists(args.output_dir)
    #         and os.listdir(args.output_dir)
    #         and args.do_train
    #         and not args.overwrite_output_dir
    # ):
    #     raise ValueError(
    #         "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
    #             args.output_dir
    #         )
    #     )

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
            args=(args, ),
            nprocs=args['distributed_training']['distributed_world_size'],
        )
    else:
        # single GPU training
        print('single GPU training...')
        single_main(args)


if __name__ == '__main__':
    cli_main()