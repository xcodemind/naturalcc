#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from ncc.data.constants import INSERTED
from dataset.codesearchnet.summarization.sum_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", type=str, default='piece', help='id(num)/piece(str)')
    parser.add_argument("--vocab-size", type=int, default=50000, help='token dictionary size')
    parser.add_argument("--src-dir", type=str,
                        default='~/.ncc/CodeSearchNet/flatten',
                        help='source data')
    parser.add_argument("--language", type=str, help='sentencepiece tokenizer for language')
    parser.add_argument("--modality", type=str, help='sentencepiece tokenizer for modality')
    parser.add_argument("--tgt-dir", type=str,
                        default='~/.ncc/CodeSearchNet/summarization/hicodebert-data-bin/wordpiece_bpe/',
                        help='save dir for sentencepiece bpe models or save files')
    parser.add_argument("--keep-empty", type=bool, default=True, help="keep empty lines")
    parser.add_argument("--overwrite", type=bool, default=False, help="build BPE model for files")
    # parser.add_argument("--insert", type=bool, help='insert CLS/S_SEP')
    parser.add_argument("--workers", type=int, default=999, help='multi-processors number')
    args = parser.parse_args()

    # parameters pre-processing
    # args.format = 'piece'
    args.language = 'ruby'
    # code, docstring, path
    args.modality = 'path'
    args.workers = min(args.workers, cpu_count())
    # args.src_dir = '~/.ncc/CodeSearchNet/flatten'
    args.src_dir = os.path.expanduser(args.src_dir)
    # args.tgt_dir = '~/.ncc/CodeSearchNet/summarization/hicodebert-data-bin/wordpiece_bpe/'
    args.tgt_dir = os.path.expanduser(args.tgt_dir)
    args.bpe_model = os.path.join(args.tgt_dir, args.language, args.modality)
    input_files = [
        os.path.join(args.src_dir, args.language, '{}.{}'.format(mode, args.modality))
        for mode in constants.MODES
    ]
    output_files = [
        os.path.join(args.tgt_dir, args.language, '{}.{}'.format(mode, args.modality))
        for mode in constants.MODES
    ]

    os.makedirs(os.path.join(args.tgt_dir, args.language), exist_ok=True)
    args.special_symbols = get_special_symbols(args)

    if args.modality == 'code':
        input_inserted_files = [in_file + INSERTED for in_file in input_files]
        for in_file, out_file in zip(input_files, input_inserted_files):
            insert_sep_token(in_file, out_file)
        input_files = input_inserted_files

    for in_file in input_files:
        if 'train' in in_file:  # only build wordpiece model on train files
            build_model(in_file, args.bpe_model, args.vocab_size, args.special_symbols)

    for input_file, output_file in zip(input_files, output_files):
        LOGGER.info('write {} into {}'.format(input_file, output_file))
        write_bpe_files(args, [input_file], [output_file])
