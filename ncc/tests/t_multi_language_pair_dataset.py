# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# import os
# import unittest
# import itertool
# import torch
# from ncc.data.multi_language_pair_dataset import MultiLanguagePairDataset
# from ncc.data.token_block_dataset import TokenBlockDataset
# from ncc.data.concat_dataset import ConcatDataset
# from ncc.tests.test_train import mock_dict
# from ncc.data import indexed_dataset
# from ncc.data import data_utils
# from ncc.data.append_token_dataset import AppendTokenDataset
# from ncc.data.truncate_dataset import TruncateDataset
# from ncc.data.strip_token_dataset import StripTokenDataset
# from ncc.data.prepend_token_dataset import PrependTokenDataset
# -*- coding: utf-8 -*-
import os
import sys
from collections import namedtuple
from ncc import LOGGER
from ncc.utils.util_file import load_yaml
from ncc.tasks.summarization import load_multimodalpair_dataset
from ncc.tasks import FairseqTask
from collections import OrderedDict
from ncc.utils import utils
from ncc import tasks
import torch
from ncc.data.multi_language_pair_dataset import collate

if __name__ == '__main__':
    Argues = namedtuple('Argues', 'yaml')

    args_ = Argues('/data/wanyao/Dropbox/ghproj-titan/naturalcodev3/run/summarization/mm2seq/ruby.yml')  # train_sl
    LOGGER.info(args_)
    # print(type(args.multi_processing))
    # assert False
    print('args: ', type(args_))
    # config = run_init(args.yaml, config=None)
    yaml_file = os.path.join(sys.path[0], args_.yaml)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)

    LOGGER.info(args)

    data_path = '/data/wanyao/ghproj_d/naturalcodev3/codesearchnet/data-bin'
    split = 'train'
    src_modalities = ['path'] # , 'code'
    # src_dicts = None
    tgt = 'docstring'
    # tgt_dict = None
    combine = False

    tgt_dict = FairseqTask.load_dictionary(os.path.join(data_path, 'dict.{}.txt'.format(args['task']['target_lang'])))
    # src_dict = {modality: None for modality in args['task']['source_lang']}
    src_dicts = OrderedDict()
    for modality in args['task']['source_lang']:
        if modality == 'path':  # special for path modality
            dict_path_border = FairseqTask.load_dictionary(
                os.path.join(data_path, 'dict.{}_border.txt'.format(modality)))  # args['task']['source_lang']
            dict_path_center = FairseqTask.load_dictionary(
                os.path.join(data_path, 'dict.{}_center.txt'.format(modality)))  # args['task']['source_lang']
            src_dicts[modality] = [dict_path_border, dict_path_center]
            # assert src_dicts[modality][0].pad() == src_dicts[modality][1].pad() == tgt_dict.pad()
            # assert src_dicts[modality][0].eos() == src_dicts[modality][1].eos() == tgt_dict.eos()
            # assert src_dicts[modality][0].unk() == src_dicts[modality][1].unk() == tgt_dict.unk()
        else:
            src_dicts[modality] = FairseqTask.load_dictionary(
                os.path.join(data_path, 'dict.{}.txt'.format(modality)))  # args['task']['source_lang']
            # assert src_dicts[modality].pad() == tgt_dict.pad()
            # assert src_dicts[modality].eos() == tgt_dict.eos()
            # assert src_dicts[modality].unk() == tgt_dict.unk()

    dataset = load_multimodalpair_dataset(
        data_path, split, src_modalities, src_dicts, tgt, tgt_dict,
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
    batch = collate(
        samples, pad_idx=src_dicts['path'][0].pad(), eos_idx=dataset.eos, src_modalities=['path'],
        left_pad_source=dataset.left_pad_source, left_pad_target=dataset.left_pad_target,
        input_feeding=dataset.input_feeding,
    )
    print(batch['net_input']['src_lengths']['path'][1])
    # sys.exit()

    # data_iter = iter(dataloader)
    # batch_data = data_iter.__next__()

    epoch_itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=args['dataset']['max_tokens'],
        max_sentences=None,     # args['dataset']['max_sentences'],
        max_positions=None,     #  args['task']['max_source_positions'],
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
    print('itr: ', itr)
    for i, obj in enumerate(itr):
        print('i: ', i)
        print('obj: ', obj)
