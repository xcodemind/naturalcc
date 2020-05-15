#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""
from typing import *

import os
import sys
from collections import OrderedDict
from collections import namedtuple

from ncc import tasks
from ncc.data import Dictionary
from ncc.utils.util_file import load_yaml
from ncc import LOGGER

from dataset.codesearchnet.summarization.sum_utils import (
    load_dict, make_all,
)


def build_vocab_dict(args: Dict, overwrite: bool = False):
    """Build vocabulary (dictionary) for source and target domain"""
    LOGGER.info('Build vocabularies...')
    task = tasks.get_task(args['preprocess']['task'])
    src_dicts = OrderedDict()
    assert args['preprocess']['trainpref'], RuntimeError('Build vocabularies from train dataset, but it is null.')
    for modal in args['preprocess']['source_lang']:
        src_dicts[modal] = load_dict(args, task, modal, overwrite)
    return src_dicts


def build_dataset(args: Dict, dicts: Dict[str, Dictionary]):
    """build dataset for modal"""
    for modal, dict in dicts.items():
        LOGGER.info('Building dataset for {}'.format(modal))
        make_all(args, modal, dict)


def main(args):
    LOGGER.info('mkdir for {} task'.format(args['preprocess']['task']))
    os.makedirs(args['preprocess']['destdir'], exist_ok=True)
    dicts = build_vocab_dict(args, overwrite=False)
    build_dataset(args, dicts)


def cli_main():
    Argues = namedtuple('Argues', 'yaml')
    args_ = Argues('preprocess_summarization.yml')  # train_sl
    LOGGER.info(args_)
    yaml_file = os.path.join(os.path.dirname(__file__), args_.yaml)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == "__main__":
    cli_main()
    sys.exit()
