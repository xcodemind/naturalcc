#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""
from typing import Dict, List

import argparse

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

from dataset.codesearchnet.utils.codebert_utils import (
    build_dictionary, make_all, make_binary_dataset,
)


def build_vocab_dict(args: Dict, overwrite: bool = False):
    """Build vocabulary (dictionary) for source and target domain"""
    LOGGER.info('Build vocabularies...')
    # task = tasks.get_task(args['preprocess']['task'])
    src_dicts = OrderedDict()

    assert args['preprocess']['joined_dictionary']

    joined_dictionary_filename = os.path.join(args['preprocess']['destdir'],
                                              '{}.dict.txt'.format('_'.join(args['preprocess']['source_lang'])))
    if os.path.exists(joined_dictionary_filename):
        joined_dictionary = Dictionary.load(joined_dictionary_filename)
    else:
        joined_dictionary = Dictionary.load(args['preprocess']['srcdict'])
        joined_dictionary.save(joined_dictionary_filename)

    for modality in args['preprocess']['source_lang']:
        src_dicts[modality] = joined_dictionary
    tgt_dict = joined_dictionary
    return src_dicts, tgt_dict


######################################################################
# dataset functions
######################################################################


def build_dataset(args: Dict, src_dicts: Dict[str, Dictionary], tgt_dict: Dictionary):
    """build dataset for modal"""
    for modality, src_dict in src_dicts.items():
        LOGGER.info('Building dataset for {}'.format(modality))
        make_all(args, modality, src_dict)
    if not args['preprocess']['only_source']:
        make_all(args, args['preprocess']['target_lang'], tgt_dict)


def main(args):
    LOGGER.info('mkdir for {} task'.format(args['preprocess']['task']))
    os.makedirs(args['preprocess']['destdir'], exist_ok=True)
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
