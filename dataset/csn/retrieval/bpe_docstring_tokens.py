# -*- coding: utf-8 -*-

import argparse

import ujson
import os
import itertools
from glob import glob
from dataset.common.codebert_utils import (
    build_model, vocab2dict,
)
from ncc import LOGGER
from ncc.data import constants
from dataset.csn import (MODES, FLATTEN_DIR)

if __name__ == '__main__':
    """
    how to run
    python -m dataset.common.bpe --src-files ~/.ncc/codenn/flatten/*.code,~/.ncc/CodeSearchNet/flatten/java/*.code
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--format", type=str, default='piece', help='id(num)/piece(str)')
    parser.add_argument("--vocab-size", type=int, default=10000, help='token dictionary size')
    parser.add_argument("--bpe-model", type=str, default='retrieval-bpe', help='BPE model and vocab name')
    # parser.add_argument("--attribute", type=str, default='joint.docstring_tokens', help='attributes')
    parser.add_argument("--attribute", type=str, default='joint.code_tokens', help='attributes')
    parser.add_argument("--language", type=str, default='javascript', help='language')
    parser.add_argument("--src-files", type=str, default=FLATTEN_DIR,
                        help='source data. E.g. *.* denotes [train/valid/test].[code/docstring]')
    parser.add_argument("--tgt-dir", type=str, default=os.path.join(FLATTEN_DIR, '*'),
                        help='save dir for sentencepiece bpe models or save files')
    parser.add_argument("--keep-empty", type=bool, default=True, help="keep empty lines")
    # parser.add_argument("--workers", type=int, default=999, help='multi-processors number')
    args = parser.parse_args()
    args.vocab_size = args.vocab_size - 1  # because sentencepiece lacks <PAD>, therefore we need to vocab_size-1
    # args.workers = min(args.workers, cpu_count())

    args.src_files = os.path.join(args.src_files, args.language, '*.{}'.format(args.attribute))
    args.src_files = [directory for directory in glob(os.path.expanduser(args.src_files)) if args.language in directory]
    args.tgt_dir = [directory for directory in glob(os.path.expanduser(args.tgt_dir)) if args.language in directory][0]
    args.bpe_model = os.path.join(args.tgt_dir, '{}-{}'.format(args.bpe_model, args.attribute))

    # # ======== STEP1 replace \n with S_SEP ======== #
    # for idx, filename in enumerate(args.src_files):
    #     filename_inserted = filename + constants.INSERTED
    #     LOGGER.info('Replace [\\n] with <S_SEP>, loading from {} and save at {}'.format(
    #         filename, filename_inserted))
    #     insert_sep_token(filename, filename_inserted)
    #     args.src_files[idx] = filename_inserted

    # ======== STEP2 merge all string into a cache file ======== #
    # only build sentencepiece model on train files
    train_files = [file for file in args.src_files if os.path.basename(file).startswith('train.')]
    LOGGER.info('Sentencepice BPE *.model and *.vocab generation, save at {}'.format(args.bpe_model))
    sp = build_model(train_files, args.bpe_model, args.vocab_size, None)

    # rewrite bpe.vocab into bpe.dict.txt so that we can use for model training and case-study
    vocab2dict(
        vocab_file='{}.vocab'.format(args.bpe_model),
        dict_file=os.path.expanduser(
            '~/.ncc/code_search_net/individual/code_tokens_docstring_tokens/data-mmap/{}/bpe.{}.dict.json'.format(
                args.language, args.attribute))
    )

    tgt_dir = os.path.dirname(args.src_files[0])
    # encode: text => id
    for mode in MODES:
        joint_docstring_file = os.path.join(FLATTEN_DIR, args.language, '{}.{}'.format(mode, args.attribute))
        tgt_file = os.path.join(tgt_dir, '{}.bpe.{}'.format(mode, args.attribute))
        tgt_file = os.path.expanduser(tgt_file)
        with open(joint_docstring_file, 'r') as reader, open(tgt_file, 'w') as writer:
            for line in reader:
                joint_docstring = ujson.loads(line)
                bpe_docstring_tokens = sp.encode_as_pieces(joint_docstring)
                print(ujson.dumps(bpe_docstring_tokens, ensure_ascii=False), file=writer)
