#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""
from typing import Dict, List

import os
import re
import ujson
import shutil
from collections import namedtuple
from multiprocessing import Pool

from ncc import tasks
from collections import (
    Counter,
    OrderedDict,
)
from ncc.data import (
    Dictionary,
    constants,
    indexed_dataset,
)
from ncc.data.binarizer import Binarizer
from ncc.utils import (
    utils, tokenizer
)
from ncc.utils.util_file import load_yaml
from ncc import LOGGER

from dataset.codesearchnet.summarization.sum_utils import *


def build_vocab_dict(args: Dict, overwrite: bool = False):
    """Build vocabulary (dictionary) for source and target domain"""
    LOGGER.info('Build vocabularies...')
    task = tasks.get_task(args['preprocess']['task'])
    src_dicts = OrderedDict()
    assert args['preprocess']['trainpref'], RuntimeError('Build vocabularies from train dataset, but it is null.')
    target = not args['preprocess']['only_source']

    for modality in args['preprocess']['source_lang']:
        src_dicts[modality] = load_dict(args, task, modality, overwrite)
    # for joined dictionary
    if args['preprocess']['joined_dictionary']:
        # the tgt_dict equals
        tgt_dict = src_dicts['code'] if 'code' in src_dicts else None
    else:
        if target:
            tgt_dict = load_dict(args, task, args['preprocess']['target_lang'], overwrite)
        else:
            tgt_dict = None
    return src_dicts, tgt_dict


######################################################################
# dataset functions
######################################################################


def make_dataset(args, vocab, input_prefix, output_prefix, lang, num_workers=1):
    if args['preprocess']['dataset_impl'] == "raw":
        # Copy original text file to destination folder
        output_text_file = dest_path(args,
                                     output_prefix,
                                     # + ".{}-{}".format(args['preprocess']['source_lang'], args['preprocess']['target_lang'])
                                     lang,
                                     )
        if lang == 'docstring':  # since docstring did't be inserted <S_SEP>, therefore the inserted should be set to False
            shutil.copyfile(file_name(input_prefix, lang, inserted=False), output_text_file)
        else:
            shutil.copyfile(file_name(input_prefix, lang, inserted=args['preprocess']['inserted']), output_text_file)
    else:
        if lang == 'docstring':
            make_binary_dataset(args, vocab, input_prefix, output_prefix, lang, num_workers, inserted=False)
        else:
            make_binary_dataset(args, vocab, input_prefix, output_prefix, lang, num_workers,
                                inserted=args['preprocess']['inserted'])


def build_dataset(args: Dict, src_dicts: Dict[str, Dictionary], tgt_dict: Dictionary):
    """build dataset for modal"""
    for modality, src_dict in src_dicts.items():
        LOGGER.info('Building dataset for {}'.format(modality))
        make_all(args, modality, src_dict)
    target = not args['preprocess']['only_source']
    if target:
        make_all(args, args['preprocess']['target_lang'], tgt_dict)


def build_insert_files(args: Dict, attr: str):
    """insert special tokens for the code modality"""
    for mode in constants.MODES:
        src_file = file_name(args['preprocess']['{}pref'.format(mode)], attr)
        tgt_file = file_name(args['preprocess']['{}pref'.format(mode)], attr, inserted=True)
        insert_sep_token(src_file, tgt_file)


def main(args):
    LOGGER.info('mkdir for {} task'.format(args['preprocess']['task']))
    os.makedirs(args['preprocess']['destdir'], exist_ok=True)
    build_insert_files(args, attr='code')
    exit()
    # 1. build vocabulary
    src_dicts, tgt_dict = build_vocab_dict(args, overwrite=True)
    # 2. build dataset
    build_dataset(args, src_dicts, tgt_dict)


def cli_main():
    Argues = namedtuple('Argues', 'yaml')
    args_ = Argues('preprocess_hicodebert.yml')  # train_sl
    LOGGER.info(args_)
    yaml_file = os.path.join(os.path.dirname(__file__), args_.yaml)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == "__main__":
    cli_main()
