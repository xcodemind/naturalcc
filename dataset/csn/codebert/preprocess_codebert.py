#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

import itertools
import shutil
from typing import Dict, List
from ncc.utils import (
    utils, tokenizer
)
import os
from multiprocessing import Pool
from collections import (namedtuple, OrderedDict, Counter)
from ncc.data import (Dictionary, indexed_dataset)
from ncc.utils.util_file import load_yaml
from ncc import tasks
from ncc.data.tools.binarizer import Binarizer

import sentencepiece as spm

from ncc import LOGGER


def binarize(args: Dict, filename: str, dict: Dictionary, out_file_prefix: str, attr: str,
             offset: int, end: int):
    """binarize function for multi-processing"""
    ds_file = '{}.mmap'.format(out_file_prefix)
    ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(dict))

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize_bpe(filename, dict, consumer, offset=offset, end=end)
    ds.finalize('{}.idx'.format(out_file_prefix))
    return res


def main(args):
    task = tasks.get_task(args['preprocess']['task'])
    LOGGER.info('mkdir for {} task'.format(args['preprocess']['task']))
    os.makedirs(args['preprocess']['destdir'], exist_ok=True)

    def load_bpe_vocab(args):
        """Build vocabulary (dictionary) for source and target domain"""
        LOGGER.info('Load BPE vocabularies...')
        # ref: https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb
        # makes segmenter instance and loads the model file (codesearchnet.model)
        bpe_dict_filename = args['preprocess']['srcdict'] + '.model'

        if args['preprocess']['dataset_impl'] == 'mmap':
            # copy (*.model) and (*.vocab) to mmap directory
            shutil.copy(args['preprocess']['srcdict'] + '.model', args['preprocess']['destdir'])
            shutil.copy(args['preprocess']['srcdict'] + '.vocab', args['preprocess']['destdir'])
            # copy (*.dict.txt) for model training. (*.dict.txt) contains same tokens of BPE dictionary
            # but it has been transformed for ncc.Dictionary class usage while training
            shutil.copy(args['preprocess']['srcdict'] + '.dict.txt', args['preprocess']['destdir'])

        bpe_dict_filename = os.path.join(args['preprocess']['destdir'], os.path.basename(bpe_dict_filename))
        sp = spm.SentencePieceProcessor()
        sp.load(bpe_dict_filename)

        src_dicts = OrderedDict()
        for modality in args['preprocess']['source_lang']:
            src_dicts[modality] = sp

        if args['preprocess']['target_lang'] is None or args['preprocess']['only_source']:
            tgt_dict = None
        else:
            tgt_dict = sp

        return src_dicts, tgt_dict

    # 1. build vocabulary from bpe directory
    src_dicts, tgt_dict = load_bpe_vocab(args)

    # 2. ***************build dataset********************
    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, lang):
        return os.path.join(args['preprocess']['destdir'], file_name(prefix, lang))

    def make_binary_dataset(dict: Dictionary, input_file, output_file,
                            attr: str, num_workers: int):
        """make binary dataset"""
        LOGGER.info("[{}] Dictionary: {} types".format(attr, len(dict) - 1))
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
                        attr,
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
            Binarizer.binarize_bpe(
                input_file, dict, lambda t: ds.add_item(t), offset=0, end=offsets[1]
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
            "[{}] {}: {} sents, {} tokens, BPE no replaced token".format(
                attr,
                input_file,
                n_seq_tok[0],
                n_seq_tok[1],
            )
        )

    def make_dataset(vocab, input_prefix, output_prefix, lang, num_workers=1):
        if args['preprocess']['dataset_impl'] == "raw":
            # because data from bpe is directly save in data-raw, no need to process
            ...
        else:
            in_file = file_name(input_prefix, lang)
            out_file = dest_path(output_prefix, lang)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            make_binary_dataset(vocab, in_file, out_file, lang, num_workers)

    def make_all(lang, vocab):
        if args['preprocess']['trainpref']:
            make_dataset(vocab, args['preprocess']['trainpref'], "train", lang,
                         num_workers=args['preprocess']['workers'])
        if args['preprocess']['validpref']:
            for k, validpref in enumerate(args['preprocess']['validpref'].split(",")):
                outprefix = "valid{}".format(k) if k > 0 else "valid"
                make_dataset(vocab, validpref, outprefix, lang, num_workers=args['preprocess']['workers'])
        if args['preprocess']['testpref']:
            for k, testpref in enumerate(args['preprocess']['testpref'].split(",")):
                outprefix = "test{}".format(k) if k > 0 else "test"
                make_dataset(vocab, testpref, outprefix, lang, num_workers=args['preprocess']['workers'])

    def build_dataset(args: Dict, src_dicts: Dict[str, Dictionary], tgt_dict: Dictionary):
        """build dataset for modal"""
        for modality, src_dict in src_dicts.items():
            LOGGER.info('Building dataset for {}'.format(modality))
            make_all(modality, src_dict)

    # 2. build dataset
    build_dataset(args, src_dicts, tgt_dict)


def cli_main():
    Argues = namedtuple('Argues', 'yaml')
    args_ = Argues('preprocess_codebert.yml')  # train_sl
    LOGGER.info(args_)
    yaml_file = os.path.join(os.path.dirname(__file__), 'config', args_.yaml)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == "__main__":
    cli_main()
