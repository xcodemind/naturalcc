# -*- coding: utf-8 -*-

import argparse

import os
import re
import ujson
import shutil
import itertools
from glob import glob
from multiprocessing import cpu_count

from dataset.common.codebert_utils import (
    insert_sep_token, build_model, get_special_symbols,
)
from ncc import LOGGER
from ncc.data import constants

CACHE_DIR = '~/.ncc/cache'
CACHE_DIR = os.path.expanduser(CACHE_DIR)
os.makedirs(CACHE_DIR, exist_ok=True)  # mkdir cache dir

if __name__ == '__main__':
    """
    how to run
    python -m dataset.common.bpe --src-files ~/.ncc/codenn/flatten/*.code,~/.ncc/CodeSearchNet/flatten/ruby/*.code
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--format", type=str, default='piece', help='id(num)/piece(str)')
    parser.add_argument("--vocab-size", type=int, default=50000, help='token dictionary size')
    parser.add_argument("--bpe-model", type=str, default='bpe', help='BPE model and vocab name')
    parser.add_argument("--src-files", type=str,
                        default='~/.ncc/codenn/flatten/*.code,~/.ncc/CodeSearchNet/flatten/ruby/*.code',
                        help='source data. E.g. *.* denotes [train/valid/test].[code/docstring]')
    parser.add_argument("--tgt-dir", type=str, default='~/.ncc/tmp',
                        help='save dir for sentencepiece bpe models or save files')
    parser.add_argument("--keep-empty", type=bool, default=True, help="keep empty lines")
    # parser.add_argument("--workers", type=int, default=999, help='multi-processors number')
    args = parser.parse_args()
    args.vocab_size = args.vocab_size - 1  # because sentencepiece lacks <PAD>, therefore we need to vocab_size-1
    # args.workers = min(args.workers, cpu_count())

    args.src_files = args.src_files.split(',')
    args.src_files = [glob(os.path.expanduser(directory)) for directory in args.src_files]
    args.src_files = list(itertools.chain(*args.src_files))
    args.tgt_dir = os.path.expanduser(args.tgt_dir)
    args.bpe_model = os.path.join(args.tgt_dir, args.bpe_model)

    # ======== STEP1 replace \n with S_SEP ======== #
    for idx, filename in enumerate(args.src_files):
        filename_inserted = filename + constants.INSERTED
        LOGGER.info('Replace [\\n] with <S_SEP>, loading from {} and save at {}'.format(
            filename, filename_inserted))
        # insert_sep_token(filename, filename_inserted)
        args.src_files[idx] = filename_inserted

    # ======== STEP2 merge all string into a cache file ======== #
    # only build sentencepiece model on train files
    train_files = [file for file in args.src_files if os.path.basename(file).startswith('train.')]
    LOGGER.info('Sentencepice BPE *.model and *.vocab generation, save at {}'.format(args.bpe_model))
    special_symbols = {constants.S_SEP, constants.CLS}
    build_model(train_files, args.bpe_model, args.vocab_size, special_symbols)
