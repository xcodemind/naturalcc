#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import os
import itertools
from multiprocessing import cpu_count

from ncc import LOGGER
from ncc.data import constants
from ncc.data.constants import INSERTED
from dataset.codesearchnet.utils.codebert_utils import (
    get_special_symbols,
    insert_sep_token,
    build_model,
    write_bpe_files,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", type=str, default='piece', help='id(num)/piece(str)')
    parser.add_argument("--vocab-size", type=int, default=50000, help='token dictionary size')
    parser.add_argument("--src-dir", type=str,
                        default='~/.ncc/CodeSearchNet/flatten',
                        help='source data')
    parser.add_argument("--language", type=str, help='sentencepiece tokenizer for language')
    parser.add_argument("--modalities", type=list, help='sentencepiece tokenizer for modalities')
    parser.add_argument("--tgt-dir", type=str,
                        default='~/.ncc/CodeSearchNet/codebert/hicodebert-data-bin/',
                        help='save dir for sentencepiece bpe models or save files')
    # parser.add_argument("--bpe-dir", type=str, default='wordpiece_bpe', help='wordpiece_bpe modal save direction')
    parser.add_argument("--keep-empty", type=bool, default=True, help="keep empty lines")
    parser.add_argument("--overwrite", type=bool, default=False, help="build BPE model for files")
    # parser.add_argument("--insert", type=bool, help='insert CLS/S_SEP')
    parser.add_argument("--workers", type=int, default=999, help='multi-processors number')
    args = parser.parse_args()

    # parameters pre-processing
    # args.format = 'piece'
    args.language = 'ruby'
    # code, docstring, path
    args.modalities = ['code',] # 'path',
    # args.modalities = ['code', 'docstring', ]
    args.workers = min(args.workers, cpu_count())
    # args.src_dir = '~/.ncc/CodeSearchNet/flatten'
    args.src_dir = os.path.expanduser(args.src_dir)
    # args.tgt_dir = '~/.ncc/CodeSearchNet/summarization/hicodebert-data-bin/'
    args.tgt_dir = os.path.expanduser(args.tgt_dir)
    args.bpe_models = os.path.join(args.tgt_dir, args.language, '_'.join(sorted(args.modalities)), 'codesearchnet')

    args.input_files = {
        modality: [
            os.path.join(args.src_dir, args.language, '{}.{}'.format(mode, modality))
            for mode in constants.MODES
        ]
        for modality in args.modalities
    }

    tgt_dir = os.path.join(args.tgt_dir, args.language, '_'.join(sorted(args.modalities)))
    os.makedirs(tgt_dir, exist_ok=True)
    args.output_files = {
        modality: [
            os.path.join(tgt_dir, '{}.{}.bpe'.format(mode, modality))
            for mode in constants.MODES
        ]
        for modality in args.modalities
    }

    args.special_symbols = get_special_symbols(args)

    if 'code' in args.modalities:
        input_inserted_files = [in_file + INSERTED for in_file in args.input_files['code']]
        for in_file, out_file in zip(args.input_files['code'], input_inserted_files):
            insert_sep_token(in_file, out_file, overwrite=True)
        args.input_files['code'] = input_inserted_files

    # only build wordpiece model on train files
    train_input_files = [file for file in itertools.chain(*args.input_files.values()) if 'train' in file]
    build_model(train_input_files, args.bpe_models, args.vocab_size, args.special_symbols)

    for modality in args.modalities:
        for input_file, output_file in zip(args.input_files[modality], args.output_files[modality]):
            LOGGER.info('write {} into {}'.format(input_file, output_file))
            write_bpe_files(args, [input_file], [output_file])
