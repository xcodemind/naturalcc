#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate the perplexity of a trained language model.
"""
import os
import sys
from collections import namedtuple
import torch
from ncc import LOGGER
from ncc import tasks
from ncc.utils import checkpoint_utils
from ncc.logging import progress_bar
from ncc.utils import utils
from ncc.utils.util_file import load_yaml
from ncc.logging.meters import StopwatchMeter, TimeMeter
import torch.nn.functional as F


def main(args):
    if args['eval']['results_path'] is not None:
        os.makedirs(args['eval']['results_path'], exist_ok=True)
        output_path = os.path.join(args['eval']['results_path'], 'generate-{}.txt'.format(args['eval']['gen_subset']))
        with open(output_path, 'w', buffering=1) as h:
            return _main(args, h)
    else:
        return _main(args, sys.stdout)


def accuracy(output, target, topk=(1,), ignore_idx=[]):
    with torch.no_grad():
        maxk = max(topk)

        # Get top predictions per position that are not in ignore_idx
        target_vocab_size = output.size(2)
        keep_idx = torch.tensor([i for i in range(target_vocab_size) if i not in ignore_idx],
                                device=output.device).long()
        _, pred = output[:, :, keep_idx].topk(maxk, 2, True, True)  # BxLx5
        pred = keep_idx[pred]  # BxLx5

        # Compute statistics over positions not labeled with an ignored idx
        correct = pred.eq(target.unsqueeze(-1).expand_as(pred)).long()
        mask = torch.ones_like(target).long()
        for idx in ignore_idx:
            mask = mask.long() & (~target.eq(idx)).long()
        mask = mask.long()
        deno = mask.sum().item()
        correct = correct * mask.unsqueeze(-1)
        res = []
        for k in topk:
            correct_k = correct[..., :k].view(-1).float().sum(0)
            res.append(correct_k.item())

        return res, deno


def _main(args, output_file):
    use_cuda = torch.cuda.is_available() and not args['common']['cpu']

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args['dataset']['test_subset'])

    # # Set dictionaries
    # try:
    #     src_dict = getattr(task, 'source_dictionary', None)
    # except NotImplementedError:
    #     src_dict = None
    # tgt_dict = task.target_dictionary

    # Load ensemble
    LOGGER.info('loading model(s) from {}'.format(args['eval']['path']))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args['eval']['path']),
        arg_overrides=eval(args['eval']['model_overrides']),
        task=task,
    )

    # Optimize ensemble for generation
    for model in models:
        # model.make_generation_fast_(
        #     beamable_mm_beam_size=None if args['eval']['no_beamable_mm'] else args['eval']['beam'],
        #     need_attn=args['eval']['print_alignment'],
        # )
        if _model_args['common']['fp16']:
            model.half()
        if use_cuda:
            model.cuda()

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args['dataset']['test_subset']),
        max_tokens=args['dataset']['max_tokens'],
        max_sentences=args['dataset']['max_sentences'],
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=_model_args['dataset']['skip_invalid_size_inputs_valid_test'],
        required_batch_size_multiple=_model_args['dataset']['required_batch_size_multiple'],
        num_shards=_model_args['dataset']['num_shards'],
        shard_id=_model_args['dataset']['shard_id'],
        num_workers=_model_args['dataset']['num_workers'],
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=_model_args['common']['log_format'],
        log_interval=_model_args['common']['log_interval'],
        default_log_format=('tqdm' if not _model_args['common']['no_progress_bar'] else 'none'),
    )

    # Initialize generator
    gen_timer = StopwatchMeter()
    type_predictor = task.build_type_predictor(models, args)

    num1, num5, num_labels_total = 0, 0, 0
    num1_any, num5_any, num_labels_any_total = 0, 0, 0

    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if 'net_input' not in sample:
            continue
        gen_timer.start()
        net_output = type_predictor.predict(models, sample)  # , prefix_tokens=prefix_tokens

        logits = net_output[0]
        labels = sample['labels']
        loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=task.target_dictionary.index('O'))
        print('loss: ', loss)
        (corr1_any, corr5_any), num_labels_any = accuracy(
            logits.cpu(), labels.cpu(), topk=(1, 5), ignore_idx=(task.target_dictionary.index('O'),))
        num1_any += corr1_any
        num5_any += corr5_any
        num_labels_any_total += num_labels_any

        (corr1, corr5), num_labels = accuracy(logits.cpu(), labels.cpu(), topk=(1, 5),
            ignore_idx=(task.target_dictionary.index('O'), task.target_dictionary.index('$any$')))
        num1 += corr1
        num5 += corr5
        num_labels_total += num_labels

        progress.log({'corr1': corr1, 'corr5': corr5})

        # LOGGER.info('corr1: {:.4f}, corr5: {:.4f}'.format(corr1/count, corr5/count))
        LOGGER.info(f"num1 {num1_any} num_labels_any_total {num_labels_any_total} avg acc1_any {num1_any / (num_labels_any_total + 1e-6) * 100:.4f}")

    acc1 = float(num1) / num_labels_total * 100
    acc5 = float(num5) / num_labels_total * 100
    acc1_any = float(num1_any) / num_labels_any_total * 100
    acc5_any = float(num5_any) / num_labels_any_total * 100

    LOGGER.info('acc1: {}\t acc5: {}\t acc1_any: {}\t acc5_any: {}'.format(acc1, acc5, acc1_any, acc5_any))


def cli_main():
    Argues = namedtuple('Argues', 'yaml')
    args_ = Argues('javascript_ft_typepredict.yml')  # train_sl
    # LOGGER.info(args_)
    yaml_file = os.path.join(os.path.dirname(__file__), 'config', args_.yaml)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)

    yaml_file_eval = os.path.join(os.path.dirname(__file__), 'config', 'javascript_typepredict.yml')
    LOGGER.info('Load arguments in {}'.format(yaml_file_eval))
    args_eval = load_yaml(yaml_file_eval)

    # Concatenate the training and evaluation arguments.
    args = {**args, **args_eval}

    LOGGER.info(args)
    main(args)


if __name__ == '__main__':
    cli_main()
