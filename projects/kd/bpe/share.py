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
import itertools
import logging
import shutil
from collections import namedtuple
from multiprocessing import Pool, cpu_count
from ncc.utils.mp_ppool import PPool
import sentencepiece as spm

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
from ncc.data.tools.binarizer import Binarizer
from ncc.utils import (
    utils, tokenizer
)
from ncc.utils.util_file import load_yaml
from ncc import LOGGER

logger = logging.getLogger(__name__)

import torch


def binarize_bpe(
    filename,
    dict,  # BPE Ditionary
    consumer,
    # append_eos=True, # sentencepiece cannot add extra <EOS>
    reverse_order=False,
    offset=0,
    end=-1,
):
    nseq, ntok = 0, 0  # nseq = sentence number, ntok = token number
    replaced = Counter()  # un-recorded tokens

    with open(filename, "r", encoding="utf-8") as f:
        f.seek(offset)
        # next(f) breaks f.tell(), hence readline() must be used
        line = safe_readline(f)
        while line:
            if end > 0 and f.tell() > end:
                break
            if str.endswith(filename, 'str') or str.endswith(filename, 'txt'):
                pass
            else:
                line = ujson.loads(line)

            ids = dict.EncodeAsIds(line)
            # print('ids: ', ids)
            if reverse_order:
                words = list(reversed(words))
            ids = torch.IntTensor(ids)

            nseq += 1
            ntok += len(ids)
            consumer(ids)
            line = f.readline()
    return {
        "nseq": nseq,
        "nunk": sum(replaced.values()),
        "ntok": ntok,
        "replaced": replaced,
    }


def binarize(args: Dict, filename: str, dict: spm, in_file: str, offset: int, end: int):
    """binarize function for multi-processing"""
    ds_file = '{}.mmap'.format(in_file)
    ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(dict))

    def consumer(tensor):
        ds.add_item(tensor)

    res = binarize_bpe(filename, dict, consumer=consumer, offset=offset, end=end)
    ds.finalize('{}.idx'.format(in_file))
    return res


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


def main(args):
    task = tasks.get_task(args['preprocess']['task'])
    LOGGER.info('mkdir for {} task'.format(args['preprocess']['task']))
    os.makedirs(args['preprocess']['destdir'], exist_ok=True)

    def build_vocab_dict(args):
        """Build vocabulary (dictionary) for source and target domain"""
        LOGGER.info('Build vocabularies...')
        src_dicts = OrderedDict()
        for modality in args['preprocess']['source_lang']:
            spm_dict = spm.SentencePieceProcessor()
            spm_dict.load(os.path.join(args['preprocess']['destdir'], f"{modality}.model"))
            src_dicts[modality] = spm_dict
            # save dictionary
            src_spm_dict = os.path.join(args['preprocess']['destdir'], f"{modality}.vocab")
            dst_spm_dict = os.path.join(args['preprocess']['destdir'], f"{modality}.dict.json")
            with open(src_spm_dict, 'r', encoding='utf8') as reader, \
                open(dst_spm_dict, 'w', encoding='utf8') as writer:
                for _ in range(4):
                    reader.readline()
                for line in reader:
                    token = line.strip().split()[0]
                    print(ujson.dumps([token, 1]), file=writer)

        tgt_dict = None
        return src_dicts, tgt_dict

    # 1. build vocabulary
    src_dicts, tgt_dict = build_vocab_dict(args)

    # 2. ***************build dataset********************
    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, impl, lang, modality):
        return os.path.join(args['preprocess']['destdir'], 'data-{}'.format(impl), lang, file_name(prefix, modality))

    def make_binary_dataset(dict: spm, input_file, output_file, attr: str, num_workers: int):
        """make binary dataset"""
        LOGGER.info("[{}] Dictionary: {} types".format(attr, len(dict)))
        n_seq_tok = [0, 0]
        replaced = Counter()  # save un-recorded tokens

        def merge_result(worker_result):
            replaced.update(worker_result["replaced"])
            n_seq_tok[0] += worker_result["nseq"]
            n_seq_tok[1] += worker_result["ntok"]

        # split a file into different parts
        # if use multi-processing, we first process 2nd to last file
        # 1.txt -> 10 processor, 0(p0)(0-99), 100(p1)(100-199), ...
        offsets = Binarizer.find_offsets(input_file, num_workers)
        pool = None
        if num_workers > 1:
            # p1-pN -> (1 bin-txt, 1 idx), (N bin-txt, N idx)
            pool = Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_file, worker_id)
                pool.apply_async(
                    binarize,
                    (
                        args,
                        input_file,
                        dict,
                        prefix,
                        offsets[worker_id],
                        offsets[worker_id + 1]
                    ),
                    callback=merge_result
                )
            pool.close()
        # process 1th file, if multi-processing available. If not, process all file
        # p0 -> 0,end
        ds_file = '{}.mmap'.format(output_file)
        ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(dict))
        merge_result(
            binarize_bpe(
                input_file, dict, consumer=lambda t: ds.add_item(t), offset=0, end=offsets[1]
            )
        )
        if num_workers > 1:
            # p1-pN
            pool.join()
            # merge sub-processors' index and data files into final files and delete them.
            for worker_id in range(1, num_workers):
                temp_file_path = "{}{}".format(output_file, worker_id)
                ds.merge_file_(temp_file_path)
                # idx, txt
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))
        ds.finalize('{}.idx'.format(output_file))

        LOGGER.info(
            "[{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
                attr,
                input_file,
                n_seq_tok[0],
                n_seq_tok[1],
                100 * sum(replaced.values()) / n_seq_tok[1],
                '[UNK]',
            )
        )

    def make_dataset(vocab, input_prefix, output_prefix, lang, modality, num_workers=1):
        if num_workers is None:
            num_workers = cpu_count()
        else:
            num_workers = min(num_workers, cpu_count())

        in_file = file_name(input_prefix, modality)
        out_file = dest_path(output_prefix, args['preprocess']['dataset_impl'], lang, modality)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        if args['preprocess']['dataset_impl'] == "raw":
            logger.info('Copying {} into {}'.format(in_file, out_file))
            shutil.copy(src=in_file, dst=out_file)
        else:
            make_binary_dataset(vocab, in_file, out_file, modality, num_workers)

    def make_all(modality, vocab, lang, data_prefs):
        num_workers = min(args['preprocess']['workers'], cpu_count())
        if data_prefs['trainpref']:
            make_dataset(vocab, data_prefs['trainpref'], "train", lang, modality, num_workers=num_workers)
        if data_prefs['validpref']:
            for k, validpref in enumerate(data_prefs['validpref'].split(",")):
                outprefix = "valid{}".format(k) if k > 0 else "valid"
                make_dataset(vocab, validpref, outprefix, lang, modality, num_workers=num_workers)
        if data_prefs['testpref']:
            for k, testpref in enumerate(data_prefs['testpref'].split(",")):
                outprefix = "test{}".format(k) if k > 0 else "test"
                make_dataset(vocab, testpref, outprefix, lang, modality, num_workers=num_workers)

    def build_dataset(args: Dict, src_dicts: Dict[str, Dictionary], tgt_dict: Dictionary):
        """build dataset for modal"""
        for modality, src_dict in src_dicts.items():
            LOGGER.info('Building dataset for {}'.format(modality))
            for lang, data_prefs in args['preprocess']['dataprefs'].items():
                make_all(modality, src_dict, lang, data_prefs)

    # 2. build dataset
    build_dataset(args, src_dicts, tgt_dict)


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(description="Generating raw/bin dataset")
    parser.add_argument(
        "--yaml_file", "-f", default='share', type=str,
        help="load {yaml_file}.yml for train",
    )
    args = parser.parse_args()
    LOGGER.info(args)
    yaml_file = os.path.join(os.path.dirname(__file__), f'{args.yaml_file}.yml')
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == "__main__":
    cli_main()
