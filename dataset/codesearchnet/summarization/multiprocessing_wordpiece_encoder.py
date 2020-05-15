#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# comand: python -m dataset.codesearchnet.summarization.multiprocessing_wordpiece_encoder --inputs ~/.ncc/CodeSearchNet/flatten/ruby/train.code --model-prefix ~/.ncc/CodeSearchNet/summarization/hicodebert-data-bin/wordpiece_bpe/codesearchnet --outputs ~/.ncc/CodeSearchNet/summarization/hicodebert-data-bin/codesearchnet.train.bpe --keep-empty --format piece --vocab-size 50000 --insert --workers 40
import argparse
import contextlib
import sys

from collections import Counter
from multiprocessing import Pool

# from ncc.data.encoders.gpt2_bpe_utils import get_encoder
import sentencepiece as spm
import ujson
import re
# S_SEP = '<S_SEP>'
# CLS = '<CLS>'
from ncc.data import constants

def build_model(input, model_prefix, vocab_size):
    spm.SentencePieceTrainer.Train('--input={} --model_prefix={} --vocab_size={} --user_defined_symbols={},{}'.format(input, model_prefix, vocab_size, constants.CLS, constants.S_SEP))
    # spm.SentencePieceTrainer.Train('--input={} --model_prefix={} --vocab_size={}'.format(input, model_prefix, vocab_size))


def insert_sep_token(input_file):
    # output_text_file = dest_path(args,
    #                              output_prefix,
    #                              # + ".{}-{}".format(args['preprocess']['source_lang'], args['preprocess']['target_lang'])
    #                              lang,
    #                              )
    with open(input_file + '_inserted', 'w') as out_file:
        with open(input_file, 'r') as in_file:
            for line in in_file.readlines():
                ln = ujson.loads(line)
                # for count in range(10, 1, -1):  # to handle duplicate '\n'
                #     ln = ln.replace('\n', S_SEP, count)
                ln = re.sub('\n\s*\n', '\n', ln)  # remove "\n \n" -> \n
                ln = ln.replace('\n', ' ' + constants.S_SEP + ' ')  # should be whitespace before and after S_SEP
                ln = constants.CLS + ' ' + ln  # profix <CLS>, should be whitespace after the CLS
                out_file.write(ujson.dumps(ln) + '\n')

def main(args):
    assert len(args.inputs) == len(args.outputs), \
        "number of input and output paths should match"

    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input, "r", encoding="utf-8"))
            if input != "-" else sys.stdin
            for input in args.inputs
        ]
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-" else sys.stdout
            for output in args.outputs
        ]

        encoder = MultiprocessingEncoder(args)
        pool = Pool(args.workers, initializer=encoder.initializer)
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
        # bpe = get_encoder(self.args.encoder_json, self.args.vocab_bpe)
        sp = spm.SentencePieceProcessor()
        sp.Load('{}.model'.format(self.args.model_prefix))

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
            if len(line) == 0 and not self.args.keep_empty:
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

    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--encoder-json",
    #     help='path to encoder.json',
    # )
    # parser.add_argument(
    #     "--vocab-bpe",
    #     type=str,
    #     help='path to vocab.bpe',
    # )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=50000,
        help='path to vocab.bpe',
    )
    parser.add_argument(
        "--format",
        type=str,
        help='path to vocab.bpe',
    )

    parser.add_argument(
        "--model-prefix",
        type=str,
        help='path to vocab.bpe',
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=['-'],
        help="input files to filter/encode",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=['-'],
        help="path to save encoded outputs",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="keep empty lines",
    )
    parser.add_argument(
        "--insert",
        action="store_true",
        help="keep empty lines",
    )
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    # sp = spm.SentencePieceProcessor()
    # sp.Load('{}.model'.format(args.model_prefix))
    # s= "13 23 24 6 3 21 237 12 203 4 204 11 5 6 7 3 537 4 204 10 2707 4 204 12 203 4 204 8 207 11 5 6 7 3 2063 4 204 12 203 4 204 11 5 6 7 3 299 5 6 7 3 528 5 6 7 3 2063 4 204 12 1069 4 204 11 5 6 7 3 15 22"
    # txt = sp.decode_ids([int(i) for i in s.split(' ')])
    # print (txt)
    # sys.exit()


    input_file = args.inputs[0]

    if args.insert:
        insert_sep_token(input_file)
    if 'train' in input_file:  # only build wordpiece model on train files
        build_model(input_file + '_inserted', args.model_prefix, args.vocab_size)
        args.inputs = [input + '_inserted' for input in args.inputs]
    # build_model(input_file, args.model_prefix, args.vocab_size)

    main(args)
