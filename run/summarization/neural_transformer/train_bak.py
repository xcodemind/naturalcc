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
    for epoch, samples in enumerate(progress):
        with metrics.aggregate('train_inner'):
            trainer._set_seed()
            trainer.model.train()
            trainer.criterion.train()
            # trainer.zero_grad()
            for i, sample in enumerate(samples):
                sample = trainer._prepare_sample(sample)
                loss = trainer.model(sample['net_input']['src_tokens'], sample['net_input']['prev_output_tokens'])
                trainer.optimizer.backward(loss)
                trainer.optimizer.step()
                trainer.zero_grad()
        if epoch % 50 == 0:
            LOGGER.info('Epoch: {}/{}, loss: {}'.format(epoch, len(progress), loss.item()))


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""

    if args['dataset']['fixed_validation_seed'] is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args['dataset']['fixed_validation_seed'])

    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
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
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                args['common']['tensorboard_logdir'] if distributed_utils.is_master(args) else None
            ),
            default_log_format=('tqdm' if not args['common']['no_progress_bar'] else 'simple'),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for sample in progress:
                trainer.valid_step(sample)

        # log validation stats
        stats = get_valid_stats(args, trainer, agg.get_smoothed_values())
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[args['checkpoint']['best_checkpoint_metric']])

    return valid_losses


def validate_official(args, trainer, task, epoch_itr, subsets):
    """Run one full official validation. Uses exact spans and same
    exact match/F1 score computation as in the SQuAD script.
    Extra arguments:
        offsets: The character start/end indices for the tokens in each context.
        texts: Map of qid --> raw text of examples context (matches offsets).
        answers: Map of qid --> list of accepted answers.
    """
    from ncc.eval.eval_utils import eval_accuracies

    def tens2sen(t, tgt_dict=None):
        sentences = []
        # loop over the batch elements
        for idx, s in enumerate(t):
            sentence = []
            for wt in s:
                word = wt if isinstance(wt, int) \
                    else wt.item()
                if word in [tgt_dict.bos_index]:
                    continue
                if word in [tgt_dict.eos_index]:
                    break
                if tgt_dict and word < len(tgt_dict):
                    sentence += [tgt_dict[word]]
                else:
                    sentence += [str(word)]

            if len(sentence) == 0:
                # NOTE: just a trick not to score empty sentence
                # this has no consequence
                sentence = [str(tgt_dict.pad_index)]

            sentence = ' '.join(sentence)
            sentences += [sentence]
        return sentences

    def replace_unknown(prediction, attn, src_raw):
        """ ?
            attn: tgt_len x src_len
        """
        tokens = prediction.split()
        for i in range(len(tokens)):
            if tokens[i] == trainer.task.tgt_dict.unk:
                _, max_index = attn[i].max(0)
                tokens[i] = src_raw[max_index.item()]
        return ' '.join(tokens)

    for subset in subsets:
        # Initialize data iterator
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
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                args['common']['tensorboard_logdir'] if distributed_utils.is_master(args) else None
            ),
            default_log_format=('tqdm' if not args['common']['no_progress_bar'] else 'simple'),
        )

        # Run through examples
        examples = 0
        sources, hypotheses, references, copy_dict = dict(), dict(), dict(), dict()
        with torch.no_grad():
            for sample in progress:
                sample = trainer._prepare_sample(sample)
                decoder_out = trainer.model.decode(
                    sample['net_input']['src_tokens'], trainer.args['model']['max_target_positions']
                )
                predictions = tens2sen(decoder_out['predictions'], trainer.task.tgt_dict)
                src_raw = [trainer.task.tgt_dict.string(line) for line in sample['target']]
                if True:
                    for i in range(len(predictions)):
                        enc_dec_attn = decoder_out['attentions'][i]
                        assert enc_dec_attn.dim() == 3
                        enc_dec_attn = enc_dec_attn.mean(1)
                        predictions[i] = replace_unknown(predictions[i], enc_dec_attn, src_raw=src_raw[i])
                targets = src_raw

                examples += len(src_raw)
                for key, pred, tgt in zip(sample['id'].tolist(), predictions, targets):
                    hypotheses[key] = [pred]
                    references[key] = tgt if isinstance(tgt, list) else [tgt]

        bleu, rouge_l, meteor = eval_accuracies(hypotheses,
                                                references,
                                                sources=None,
                                                filename=None,
                                                mode='dev')
        result = dict()
        result['bleu'] = bleu
        result['rouge_l'] = rouge_l
        result['meteor'] = meteor
        LOGGER.info('dev valid official: Epoch = | bleu = %.2f | rouge_l = %.2f | ' % (bleu, rouge_l))


def get_valid_stats(args, trainer, stats):
    if 'nll_loss' in stats and 'ppl' not in stats:
        stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        key = 'best_{0}'.format(args['checkpoint']['best_checkpoint_metric'])
        best_function = max if args['checkpoint']['maximize_best_checkpoint_metric'] else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[args['checkpoint']['best_checkpoint_metric']],
        )
    return stats


def get_training_stats(stats):
    if 'nll_loss' in stats and 'ppl' not in stats:
        stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
    stats['wall'] = round(metrics.get_meter('default', 'wall').elapsed_time, 0)
    return stats


def should_stop_early(args, valid_loss):
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if args['checkpoint']['patience'] <= 0:
        return False

    def is_better(a, b):
        return a > b if args['checkpoint']['maximize_best_checkpoint_metric'] else a < b

    prev_best = getattr(should_stop_early, 'best', None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= args['checkpoint']['patience']:
            LOGGER.info('early stop since valid performance hasn\'t improved for last {} runs'.format(
                args['checkpoint']['patience']))

        return should_stop_early.num_runs >= args['checkpoint']['patience']


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
    from ncc.models.summarization.neural_transformer import NeuralTransformer
    model = NeuralTransformer(args, task.src_dict, task.tgt_dict)
    criterion = task.build_criterion(args)
    LOGGER.info(model)
    LOGGER.info('model {}, criterion {}'.format(args['model']['arch'], criterion.__class__.__name__))
    LOGGER.info('num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # 4. Build trainer
    trainer = Trainer(args, task, model, criterion)
    LOGGER.info('training on {} GPUs'.format(args['distributed_training']['distributed_world_size']))
    LOGGER.info('max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args['dataset']['max_tokens'],
        args['dataset']['max_sentences'],
    ))

    # 5. Load the latest checkpoint if one is available and restore the corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer, combine=False)

    # 6. Train until the learning rate gets too small
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
        train(args, trainer, task, epoch_itr)
        validate_official(args, trainer, task, epoch_itr, valid_subsets)
        valid_losses = [None]

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # early stop
        if should_stop_early(args, valid_losses[0]):
            LOGGER.info('early stop since valid performance hasn\'t improved for last {} runs'.format(
                args['checkpoint']['patience']))
            break

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            combine=False,  # TODO to be checked
            # sharded data: get train iterator for next epoch
            load_dataset=(os.pathsep in args['task']['data']),
        )

    train_meter.stop()
    LOGGER.info('done training in {:.1f} seconds'.format(train_meter.sum))


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
        "--language", "-l", default='python', type=str, help="load {language}.yml for train",
    )
    args = parser.parse_args()
    # Argues = namedtuple('Argues', 'yaml')
    # args_ = Argues('ruby.yml')
    yaml_file = os.path.join(os.path.dirname(__file__), 'config', '{}.yml'.format(args.language))
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
    """
    nohup python -m run.summarization.neural_transformer.train > transformer.log 2>&1 &
    """
    cli_main()
