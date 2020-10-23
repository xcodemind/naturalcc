# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
from collections import namedtuple
from ncc import LOGGER
from ncc.utils.util_file import load_yaml
from ncc.tasks.contracode import load_augmented_code_dataset
from ncc.tasks import NccTask
from ncc import tasks
import torch
from ncc.data.contracode.contracode_dataset import collate
from ncc.data import iterators
from ncc.logging import metrics, progress_bar
from ncc.utils import checkpoint_utils, distributed_utils
from ncc.trainer.ncc_trainer import Trainer
from ncc.data import constants
import numpy as np
import random


if __name__ == '__main__':
    Argues = namedtuple('Argues', 'yaml')

    args_ = Argues('javascript.yml')  # train_sl
    LOGGER.info(args_)
    # print(type(args.multi_processing))
    # assert False
    print('args: ', type(args_))
    # config = run_init(args.yaml, config=None)
    # yaml_file = os.path.join('/data/wanyao/Dropbox/ghproj-titan/naturalcodev3/run/codebert/code_roberta/', args_.yaml)
    yaml_file = os.path.join('../../../naturalcodev3/run/contracode/', 'config', args_.yaml)
    yaml_file = os.path.realpath(yaml_file)
    # yaml_file = os.path.join('/data/wanyao/Dropbox/ghproj-titan/naturalcodev3/run/summarization/seq2seq/', args_.yaml)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)

    torch.manual_seed(args['common']['seed'])
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

    data_path = os.path.expanduser('~/.ncc/augmented_javascript/contracode/data-raw')
    split = 'test'
    # src_modalities = ['path'] # , 'code'
    # src_dicts = None
    # tgt = 'docstring'
    # tgt_dict = None
    combine = False
    src_dict = task.source_dictionary
    # sp = task.sp
    # src_dict = task.load_dictionary(os.path.join(data_path, 'csnjs_8k_9995p_unigram_url.dict.txt'))

    # src_dict = NccTask.load_dictionary(
    #     os.path.join(data_path, 'dict.{}.txt'.format(args['task']['source_lang'])))  # args['task']['source_lang']

    # src_dict.add_symbol(constants.S_SEP)
    # src_dict.add_symbol(constants.S2S_SEP)
    # src_dict.add_symbol(constants.CLS)
    # src_dict.add_symbol(constants.MASK)

    # tgt_dict.add_symbol(constants.S2S_BOS)
    # tgt_dict.add_symbol(constants.T_MASK)
    # print('<T_MASK> id is', src_dict.index('<T_MASK>'))
    # print('<T_MASK> id is', tgt_dict.index('<T_MASK>'))

    # assert src_dict.pad() == tgt_dict.pad()
    # assert src_dict.eos() == tgt_dict.eos()
    # assert src_dict.unk() == tgt_dict.unk()

    # dataset = load_masked_code_docstring_dataset_unilm(
    #     data_path, split, args['task']['source_lang'], src_dict, args['task']['target_lang'], tgt_dict,
    #     combine=combine, dataset_impl=args['dataset']['dataset_impl'],
    #     upsample_primary=args['task']['upsample_primary'],
    #     left_pad_source=args['task']['left_pad_source'],
    #     left_pad_target=args['task']['left_pad_target'],
    #     max_source_positions=args['task']['max_source_positions'],
    #     max_target_positions=args['task']['max_target_positions'],
    #     load_alignments=args['task']['load_alignments'],
    #     truncate_source=args['task']['truncate_source'],
    # )
    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    # indices = dataset.ordered_indices()
    #
    # sys.exit()

    epoch = 1
    dataset = load_augmented_code_dataset(args, epoch, data_path, split, src_dict, combine)

    data_item = dataset.__getitem__(4)
    print('data_item: ', data_item)
    # exit()
    samples_0 = []
    for i in [0, 1, 2, 3]:#range(100): [4,5,6,7], [8,9,10,11]
        print('i: ', i)
        data_item = dataset.__getitem__(i)
        samples_0.append(data_item)
    batch_0 = collate(
        samples_0, src_dict=src_dict, program_mode='contrastive',
        left_pad_source=dataset.left_pad_source, left_pad_target=dataset.left_pad_target,
        # input_feeding=dataset.input_feeding,
    )
    log_output = trainer.train_step([batch_0])
    print('log_output: ', log_output)
    # ====
    samples_1 = []
    for i in [4, 5, 6, 7]:  # range(100): [4,5,6,7], [8,9,10,11]
        print('i: ', i)
        data_item = dataset.__getitem__(i)
        samples_1.append(data_item)
    batch_1 = collate(
        samples_1, src_dict=src_dict, program_mode='contrastive',
        left_pad_source=dataset.left_pad_source, left_pad_target=dataset.left_pad_target,
        # input_feeding=dataset.input_feeding,
    )
    log_output = trainer.train_step([batch_1])
    print('log_output: ', log_output)
    # ======
    samples_2 = []
    for i in [8, 9, 10, 11]:  # range(100): [4,5,6,7], [8,9,10,11]
        print('i: ', i)
        data_item = dataset.__getitem__(i)
        samples_2.append(data_item)
    batch_2 = collate(
        samples_2, src_dict=src_dict, program_mode='contrastive',
        left_pad_source=dataset.left_pad_source, left_pad_target=dataset.left_pad_target,
        # input_feeding=dataset.input_feeding,
    )
    log_output = trainer.train_step([batch_2])
    print('log_output: ', log_output)
    # =====
    samples_3 = []
    for i in [12, 13, 14, 15]:  # range(100): [4,5,6,7], [8,9,10,11]
        print('i: ', i)
        data_item = dataset.__getitem__(i)
        samples_3.append(data_item)
    batch_3 = collate(
        samples_3, src_dict=src_dict, program_mode='contrastive',
        left_pad_source=dataset.left_pad_source, left_pad_target=dataset.left_pad_target,
        # input_feeding=dataset.input_feeding,
    )
    log_output = trainer.train_step([batch_3])
    print('log_output: ', log_output)
    exit()


    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=dataset.collater,
        # batch_sampler=batches[offset:],
        num_workers=1  # args['dataset']['num_workers'],
    )


    # exit()
    # batch = collate(
    #     samples, src_dict, tgt_dict,
    #     left_pad_source=dataset.left_pad_source, left_pad_target=dataset.left_pad_target,
    #     # input_feeding=dataset.input_feeding,
    # )
    # # torch.cuda.set_device(0)
    # # batch.cuda()
    # # model.cuda()
    # print(batch)
    # # model(*batch)
    # # 'src_tokens': input_ids,
    # # 'segment_labels': segment_ids,
    # # 'attention_mask': input_mask,
    # # model(batch['net_input']['src_tokens'].cuda(), batch['net_input']['segment_labels'].cuda(), batch['net_input']['attention_mask'].cuda())
    # model(batch['net_input']['src_tokens'], batch['net_input']['segment_labels'], batch['net_input']['attention_mask'])
    # sys.exit()

    # data_iter = iter(dataloader)
    # batch_data = data_iter.__next__()
    # batch_data = data_iter.__next__()
    # print('batch_data: ', batch_data)
    # exit()
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

    # log_output = trainer.train_step([batch])
    # print('log_output: ', log_output)
    for samples in progress:
        print('samples: ', samples)
        # exit()
        log_output = trainer.train_step(samples)
        print('log_output: ', log_output)
