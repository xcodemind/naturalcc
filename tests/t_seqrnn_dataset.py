# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
from collections import namedtuple
from ncc import LOGGER
from ncc.utils.util_file import load_yaml
from ncc.tasks.completion.completion import load_tok_dataset
from ncc.tasks import NccTask
from ncc import tasks
import torch
from ncc.data.completion.seqrnn_dataset import collate
from ncc.data import iterators
from ncc.logging import metrics, progress_bar
from ncc.utils import checkpoint_utils, distributed_utils
from ncc.trainer.ncc_trainer import Trainer
from ncc.data import constants

if __name__ == '__main__':
    Argues = namedtuple('Argues', 'yaml')

    args_ = Argues('ruby.yml')  # train_sl
    LOGGER.info(args_)
    # print(type(args.multi_processing))
    # assert False
    print('args: ', type(args_))
    # config = run_init(args.yaml, config=None)
    # yaml_file = os.path.join('/data/wanyao/Dropbox/ghproj-titan/naturalcodev3/run/codebert/code_roberta/', args_.yaml)
    yaml_file = os.path.join('../../../naturalcodev3/run/completion/seqrnn/config', args_.yaml)
    yaml_file = os.path.realpath(yaml_file)
    # yaml_file = os.path.join('/data/wanyao/Dropbox/ghproj-titan/naturalcodev3/run/summarization/seq2seq/', args_.yaml)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)

    # data_path = os.path.expanduser('~/.ncc/CodeSearchNet/summarization/hicodebert-data-bin')
    data_path = os.path.expanduser('~/.ncc/py150/seqrnn_debug/data-raw')
    split = 'train'
    # src_modalities = ['path'] # , 'code'
    # src_dicts = None
    # tgt = 'docstring'
    # tgt_dict = None
    combine = False

    # tgt_dict = NccTask.load_dictionary(os.path.join(data_path, 'dict.{}.txt'.format(args['task']['target_lang'])))

    src_dict = NccTask.load_dictionary(
        os.path.join(data_path, 'dict.{}.txt'.format(args['task']['source_lang'])))  # args['task']['source_lang']

    dataset = load_tok_dataset(data_path, split, args['task']['source_lang'], src_dict,
                                                    dataset_impl=args['dataset']['dataset_impl'])
    data_item = dataset.__getitem__(0)
    print('data_item: ', data_item)
    # sys.exit()
    samples = []
    for i in range(100):
        print('i: ', i)
        data_item = dataset.__getitem__(i)
        samples.append(data_item)
    # print('samples: ', samples)
    # sys.exit()
    # sys.exit()
    task = tasks.setup_task(args)  # task.tokenizer
    model = task.build_model(args)  # , config
    criterion = task.build_criterion(args)
    LOGGER.info(model)
    LOGGER.info('model {}, criterion {}'.format(args['model']['arch'], criterion.__class__.__name__))
    LOGGER.info('num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    # indices = dataset.ordered_indices()
    #
    # sys.exit()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=dataset.collater,
        # batch_sampler=batches[offset:],
        num_workers=1  # args['dataset']['num_workers'],
    )

    # batch = collate(
    #     samples, pad_idx=src_dict.pad(), eos_idx=dataset.eos,
    #     left_pad_source=dataset.left_pad_source, left_pad_target=dataset.left_pad_target,
    #     input_feeding=dataset.input_feeding,
    # )
    # batch = collate(
    #     samples, src_dict, tgt_dict,
    #     left_pad_source=dataset.left_pad_source, left_pad_target=dataset.left_pad_target,
    #     # input_feeding=dataset.input_feeding,
    # )
    batch = collate(samples, pad_idx = src_dict.pad(), eos_idx=None)
    # torch.cuda.set_device(0)
    # batch.cuda()
    # model.cuda()
    print(batch)

    # model(*batch)
    # 'src_tokens': input_ids,
    # 'segment_labels': segment_ids,
    # 'attention_mask': input_mask,
    # model(batch['net_input']['src_tokens'].cuda(), batch['net_input']['segment_labels'].cuda(), batch['net_input']['attention_mask'].cuda())
    # model(batch['net_input']['src_tokens'], batch['net_input']['segment_labels'], batch['net_input']['attention_mask'])
    # sys.exit()

    # data_iter = iter(dataloader)
    # batch_data = data_iter.__next__()

    epoch_itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=args['dataset']['max_tokens'],
        max_sentences=args['dataset']['max_sentences'],
        max_positions=None,  # args['task']['max_source_positions'],
        ignore_invalid_inputs=True,
        required_batch_size_multiple=args['dataset']['required_batch_size_multiple'],
        seed=args['common']['seed'],
        num_shards=1,
        shard_id=0,
        num_workers=args['dataset']['num_workers'],
        epoch=1,
    )

    # itr = epoch_itr.next_epoch_itr(
    #     fix_batches_to_gpus=args['distributed_training']['fix_batches_to_gpus'],
    #     shuffle=(epoch_itr.next_epoch_idx > args['dataset']['curriculum']),
    # )
    # print('itr: ', itr)
    # for i, obj in enumerate(itr):
    #     print('i: ', i)
    #     print('obj: ', obj)

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

    for samples in progress:
        print('samples: ')
        log_output = trainer.train_step(samples)
        print('log_output: ', log_output)
