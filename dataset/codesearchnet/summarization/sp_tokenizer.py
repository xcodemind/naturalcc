#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# comand: python -m dataset.codesearchnet.summarization.multiprocessing_wordpiece_encoder --inputs ~/.ncc/CodeSearchNet/flatten/ruby/train.code --model-prefix ~/.ncc/CodeSearchNet/summarization/hicodebert-data-bin/wordpiece_bpe/codesearchnet --outputs ~/.ncc/CodeSearchNet/summarization/hicodebert-data-bin/codesearchnet.train.bpe --keep-empty --format piece --vocab-size 50000 --insert --workers 40

from typing import *

import os
import re
import sys
import ujson
import argparse
import contextlib
import sentencepiece as spm
from collections import Counter
from multiprocessing import Pool, cpu_count
from ncc.data import constants
from dataset.codesearchnet.summarization.sum_utils import *


def get_special_symbols(args: Dict) -> List:
    def _special_symbols(args: Dict) -> Optional[Set]:
        """some modality need special symbols"""
        special_symbols = set([constants.CLS])
        if args.modality in ['code']:
            special_symbols.update([constants.S_SEP])
        elif args.modality in ['path']:
            special_symbols.update([constants.H_SEP, constants.T_SEP])
            special_symbols.update(path_special_symbols(args.input_files))
        else:
            return None
        return special_symbols

    default_special_symbols_files = os.path.join(args.tgt_dir, args.language, '.{}.ss'.format(args.modality))
    if os.path.exists(default_special_symbols_files) and (not args.overwrite):
        with open(default_special_symbols_files, 'r') as reader:
            special_symbols = [line.rstrip('\n') for line in reader.readlines()]
    else:
        special_symbols = _special_symbols(args)
        with open(default_special_symbols_files, 'w') as writer:
            for symbol in special_symbols:
                writer.write(symbol + '\n')
    return special_symbols


def combine_special_symbols(tokens: List, special_symbols: Optional[Set]) -> List:
    """merge _ and special_symbols, e.g. _ <CLS> => _<CLS>"""
    new_tokens = []
    idx = 0
    while idx < len(tokens) - 1:
        if (tokens[idx] == constants.SP_SPACE) and (tokens[idx + 1] in special_symbols):
            new_tokens.append(tokens[idx] + tokens[idx + 1])
            idx += 2
        else:
            new_tokens.append(tokens[idx])
            idx += 1
    if idx == len(tokens):
        new_tokens.append(tokens[-1])
    return new_tokens


def build_model(file: str, model_name: str, vocab_size: int, special_symbols: Optional[Set] = None,
                overwrite: bool = False):
    os.makedirs(os.path.dirname(model_name), exist_ok=True)
    if os.path.exists('{}.model'.format(model_name)) and os.path.exists('{}.vocab'.format(model_name)) \
            and not overwrite:
        return
    params = '--input={} --model_prefix={} --vocab_size={} --hard_vocab_limit=false'.format(
        file, model_name, vocab_size)
    if special_symbols is not None:
        params += ' --user_defined_symbols={}'.format(','.join(special_symbols))
    spm.SentencePieceTrainer.Train(params)


def insert_sep_token(input_file: str, output_file: Optional[str] = None, overwrite: bool = False):
    """insert CLS/S_SEP for bert"""
    if output_file is None:
        input_file = input_file + '_inserted'
    if os.path.exists(output_file) and not overwrite:
        return
    with open(output_file, 'w') as writer:
        with open(input_file, 'r') as reader:
            for line in reader.readlines():
                ln = ujson.loads(line)
                ln = re.sub('\n\s*\n', '\n', ln)  # remove "\n \n" -> \n
                ln = ln.replace('\n', ' ' + constants.S_SEP + ' ')  # should be whitespace before and after S_SEP
                ln = constants.CLS + ' ' + ln  # profix <CLS>, should be whitespace after the CLS
                writer.write(ujson.dumps(ln) + '\n')


class MultiprocessingEncoder(object):

    def __init__(self, args):
        self.args = args

    def initializer(self):
        global sp
        # bpe = get_encoder(self.args.encoder_json, self.args.vocab_bpe)
        sp = spm.SentencePieceProcessor()
        sp.Load('{}.model'.format(self.args.bpe_model))

    def encode(self, line):
        global sp
        if self.args.format == 'id':
            ids = sp.EncodeAsIds(line)
            return list(map(str, ids))
        elif self.args.format == 'piece':
            pieces = sp.EncodeAsPieces(line)
            return pieces

    def decode(self, tokens):
        global sp
        # return bpe.decode(tokens)
        if self.args.format == 'id':
            return sp.DecodeIds(tokens)
        elif self.args.format == 'piece':
            return sp.DecodePieces(tokens)

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = line.strip()
            line = ujson.loads(line)
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None]
            tokens = self.encode(line)
            tokens = combine_special_symbols(tokens, self.args.special_symbols)
            enc_lines.append(" ".join(tokens))
        return ["PASS", enc_lines]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]


def main(args: Dict):
    assert len(args.input_files) == len(args.output_files), \
        RuntimeError("number of input and output paths should match")

    with contextlib.ExitStack() as stack:
        input_files = [
            stack.enter_context(open(input, "r", encoding="utf-8"))
            if input != "-" else sys.stdin
            for input in args.input_files
        ]
        output_files = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-" else sys.stdout
            for output in args.output_files
        ]

        encoder = MultiprocessingEncoder(args)
        pool = Pool(args.workers, initializer=encoder.initializer)
        encoded_lines = pool.imap(encoder.encode_lines, zip(*input_files), 100)

        stats = Counter()
        for i, (filt, enc_lines) in enumerate(encoded_lines, start=1):
            if filt == "PASS":
                for enc_line, output_h in zip(enc_lines, output_files):
                    print(enc_line, file=output_h)
            else:
                stats["num_filtered_" + filt] += 1
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)


'''
python -m dataset.codesearchnet.summarization.multiprocessing_wordpiece_encoder --inputs ~/.ncc/CodeSearchNet/flatten/ruby/train.code --model-prefix ~/.ncc/CodeSearchNet/summarization/hicodebert-data-bin/wordpiece_bpe/codesearchnet --outputs ~/.ncc/CodeSearchNet/summarization/hicodebert-data-bin/codesearchnet.train.bpe --keep-empty --format piece --vocab-size 50000 --insert --workers 40
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", type=str, help='id(num)/piece(str)')
    parser.add_argument("--vocab-size", type=int, default=50000, help='token dictionary size')
    parser.add_argument("--src-dir", type=str, help='source data')
    parser.add_argument("--language", type=str, help='sentencepiece tokenizer for language')
    parser.add_argument("--modality", type=str, help='sentencepiece tokenizer for modality')
    parser.add_argument("--tgt-dir", type=str, help='save dir for sentencepiece bpe models or save files')
    parser.add_argument("--keep-empty", type=bool, default=True, help="keep empty lines")
    parser.add_argument("--overwrite", type=bool, default=False, help="build BPE model for files")
    parser.add_argument("--insert", type=bool, help='insert CLS/S_SEP')
    parser.add_argument("--workers", type=int, default=999, help='multi-processors number')
    args = parser.parse_args()

    # parameters pre-processing
    args.format = 'piece'
    args.language = 'ruby'
    args.modality = 'path'
    args.workers = min(args.workers, cpu_count())
    args.src_dir = '~/.ncc/CodeSearchNet/flatten'
    args.src_dir = os.path.expanduser(args.src_dir)
    args.tgt_dir = '~/.ncc/CodeSearchNet/summarization/hicodebert-data-bin'
    args.tgt_dir = os.path.expanduser(args.tgt_dir)
    args.bpe_model = os.path.join(args.tgt_dir, args.language, 'BPE_models', args.modality)
    args.input_files = [
        os.path.join(args.src_dir, args.language, '{}.{}'.format(mode, args.modality))
        for mode in constants.MODES
    ]
    args.output_files = [
        os.path.join(args.tgt_dir, args.language, '{}.{}'.format(mode, args.modality))
        for mode in constants.MODES
    ]
    args.insert = False

    args.special_symbols = get_special_symbols(args)
    if args.insert:
        input_inserted_files = [in_file + '_inserted' for in_file in args.input_files]
        for in_file, out_file in zip(args.input_files, input_inserted_files):
            insert_sep_token(in_file, out_file)
            if 'train' in in_file:  # only build wordpiece model on train files
                build_model(out_file, args.bpe_model, args.vocab_size, args.special_symbols)
        args.input_files = input_inserted_files
    else:
        for in_file in args.input_files:
            if 'train' in in_file:  # only build wordpiece model on train files
                build_model(in_file, args.bpe_model, args.vocab_size, args.special_symbols)
    main(args)
