#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import contextlib
import sys
import os
from collections import Counter
from multiprocessing import Pool
from collections import namedtuple
import sentencepiece as spm
import ujson
import re
from ncc.data import constants
from ncc import LOGGER
from ncc.utils.util_file import load_yaml

INSERTED_SUFFIX = '_inserted'


def build_model(input, model_prefix, vocab_size):
    spm.SentencePieceTrainer.Train('--input={} --model_prefix={} --vocab_size={} --user_defined_symbols={},{}'.format(input, model_prefix, vocab_size, constants.CLS, constants.S_SEP))


def insert_sep_token(input_file, inserted_suffix):
    # print('input_file + inserted_suffix: ', type(input_file + inserted_suffix))
    with open(input_file + inserted_suffix, 'w') as out_file:
        with open(input_file, 'r') as in_file:
            for line in in_file.readlines():
                ln = ujson.loads(line)
                ln = re.sub('\n\s*\n', '\n', ln)  # remove "\n \n" -> \n
                ln = ln.replace('\n', ' ' + constants.S_SEP + ' ')  # should be whitespace before and after S_SEP
                ln = constants.CLS + ' ' + ln  # profix <CLS>, should be whitespace after the CLS
                out_file.write(ujson.dumps(ln) + '\n')


def main(args):
    assert len(args['preprocess']['inputs']) == len(args['preprocess']['outputs']), \
        "number of input and output paths should match"

    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input, "r", encoding="utf-8"))
            if input != "-" else sys.stdin
            for input in args['preprocess']['inputs']
        ]
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-" else sys.stdout
            for output in args['preprocess']['outputs']
        ]

        encoder = MultiprocessingEncoder(args)
        pool = Pool(args['preprocess']['workers'], initializer=encoder.initializer)
        encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 100)

        stats = Counter()
        for i, (filt, enc_lines) in enumerate(encoded_lines, start=1):
            if filt == "PASS":
                for enc_line, output_h in zip(enc_lines, outputs):
                    print(enc_line, file=output_h)
            else:
                stats["num_filtered_" + filt] += 1
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)


class MultiprocessingEncoder(object):

    def __init__(self, args):
        self.args = args

    def initializer(self):
        global sp
        sp = spm.SentencePieceProcessor()
        sp.Load('{}.model'.format(self.args['preprocess']['model_prefix']))

    def encode(self, line):
        global sp
        if self.args['preprocess']['format'] == 'id':
            ids = sp.EncodeAsIds(line)
            return list(map(str, ids))
        elif self.args['preprocess']['format'] == 'piece':
            pieces = sp.EncodeAsPieces(line)
            return pieces

    def decode(self, tokens):
        global sp
        # return bpe.decode(tokens)
        if self.args['preprocess']['format'] == 'id':
            return sp.DecodeIds(tokens)
        elif self.args['preprocess']['format'] == 'piece':
            return sp.DecodePieces(tokens)

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = line.strip()
            line = ujson.loads(line)
            if len(line) == 0 and not self.args['preprocess']['keep_empty']:
                return ["EMPTY", None]
            tokens = self.encode(line)
            enc_lines.append(" ".join(tokens))
        return ["PASS", enc_lines]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]


if __name__ == "__main__":
    Argues = namedtuple('Argues', 'yaml')
    args_ = Argues('multiprocessing_wordpiece_encoder.yml')  # train_sl
    LOGGER.info(args_)
    yaml_file = os.path.join(os.path.dirname(__file__), args_.yaml)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)

    #  insert special tokens
    if args['preprocess']['insert']:
        for i, input_file in enumerate(args['preprocess']['inputs']):
            if 'docstring' not in input_file:   # only for code, not for docstrings.
                insert_sep_token(input_file, INSERTED_SUFFIX)
                args['preprocess']['inputs'][i] = input_file + INSERTED_SUFFIX
                # args['preprocess']['inputs'] = [input + INSERTED_SUFFIX for input in args['preprocess']['inputs']]

    # build model
    if args['preprocess']['train_model']:
    # for input_file in args['preprocess']['inputs']:

        # if 'train' in input_file:  # only build wordpiece model on train files
        build_model(','.join(args['preprocess']['inputs']), args['preprocess']['model_prefix'], args['preprocess']['vocab_size'])

    main(args)
