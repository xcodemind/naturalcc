# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
from collections import namedtuple
from ncc import LOGGER
from ncc.utils.util_file import load_yaml
from ncc.tasks.hi_transformer_summarization import load_codepair_dataset
from ncc.tasks import FairseqTask
from ncc import tasks
import torch
from ncc.data.summarization.language_pair_dataset import collate
from ncc.data import iterators
from ncc.logging import metrics, progress_bar
from ncc.utils import checkpoint_utils, distributed_utils
from ncc.trainer.fair_trainer import Trainer


if __name__ == '__main__':
    Argues = namedtuple('Argues', 'yaml')

    args_ = Argues('/data/wanyao/Dropbox/ghproj-titan/naturalcodev3/run/codebert/hi_codebert/ruby.yml')  # train_sl
    LOGGER.info(args_)
    # print(type(args.multi_processing))
    # assert False
    print('args: ', type(args_))
    # config = run_init(args.yaml, config=None)
    yaml_file = os.path.join(sys.path[0], args_.yaml)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)

    modality = args['task']['source_lang']

    data_path = '/data/wanyao/ghproj_d/naturalcodev3/codesearchnet/summarization/hicodebert-data-bin'
    split = 'train'
    # src_modalities = ['path'] # , 'code'
    # src_dicts = None
    tgt = 'docstring'
    # tgt_dict = None
    combine = False

    tgt_dict = FairseqTask.load_dictionary(os.path.join(data_path, 'dict.{}.txt'.format(args['task']['target_lang'])))
    # src_dict = {modality: None for modality in args['task']['source_lang']}
    # src_dicts = OrderedDict()
    if modality == 'path':  # special for path modality
        dict_path_border = FairseqTask.load_dictionary(
            os.path.join(data_path, 'dict.{}_border.txt'.format(modality)))  # args['task']['source_lang']
        dict_path_center = FairseqTask.load_dictionary(
            os.path.join(data_path, 'dict.{}_center.txt'.format(modality)))  # args['task']['source_lang']
        src_dict = [dict_path_border, dict_path_center]
        # assert src_dicts[modality][0].pad() == src_dicts[modality][1].pad() == tgt_dict.pad()
        # assert src_dicts[modality][0].eos() == src_dicts[modality][1].eos() == tgt_dict.eos()
        # assert src_dicts[modality][0].unk() == src_dicts[modality][1].unk() == tgt_dict.unk()
    else:
        src_dict = FairseqTask.load_dictionary(
            os.path.join(data_path, 'dict.{}.txt'.format(modality)))  # args['task']['source_lang']
        # assert src_dicts[modality].pad() == tgt_dict.pad()
        # assert src_dicts[modality].eos() == tgt_dict.eos()
        # assert src_dicts[modality].unk() == tgt_dict.unk()

    dataset = load_codepair_dataset(
        data_path, split, modality, src_dict, tgt, tgt_dict,
        combine=combine, dataset_impl=args['dataset']['dataset_impl'],
        upsample_primary=args['task']['upsample_primary'],
        left_pad_source=args['task']['left_pad_source'],
        left_pad_target=args['task']['left_pad_target'],
        max_source_positions=args['task']['max_source_positions'],
        max_target_positions=args['task']['max_target_positions'],
        load_alignments=args['task']['load_alignments'],
        truncate_source=args['task']['truncate_source'],
    )
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
        num_workers=args['dataset']['num_workers'],
    )
    samples = []
    for i in range(100):
        data_item = dataset.__getitem__(i)
        samples.append(data_item)
    # print('samples: ', samples)
    # sys.exit()

    # batch = collate(
    #     samples, pad_idx=src_dict.pad(), eos_idx=dataset.eos,
    #     left_pad_source=dataset.left_pad_source, left_pad_target=dataset.left_pad_target,
    #     input_feeding=dataset.input_feeding,
    # )
    batch = collate(
        samples, src_dict, tgt_dict,
        left_pad_source=dataset.left_pad_source, left_pad_target=dataset.left_pad_target,
        # input_feeding=dataset.input_feeding,
    )
    # print(batch)
    # sys.exit()

    # data_iter = iter(dataloader)
    # batch_data = data_iter.__next__()

    epoch_itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=args['dataset']['max_tokens'],
        max_sentences=args['dataset']['max_sentences'],
        max_positions=None, #args['task']['max_source_positions'],
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

