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


def binarize(args: Dict, filename: str, dict: Dictionary, in_file: str, attr: str,
             offset: int, end: int, append_eos: bool = False):
    """binarize function for multi-processing"""
    ds_file = '{}.mmap'.format(in_file)
    ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(dict))

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize(filename, dict, consumer, tokenize=tokenizer.tokenize_list,
                             append_eos=append_eos, offset=offset, end=end)
    ds.finalize('{}.idx'.format(in_file))
    return res


def main(args):
    task = tasks.get_task(args['preprocess']['task'])
    LOGGER.info('mkdir for {} task'.format(args['preprocess']['task']))
    os.makedirs(args['preprocess']['destdir'], exist_ok=True)

    def train_path(lang):
        return "{}{}".format(args['preprocess']['trainpref'], ("." + lang) if lang else "")

    def build_dictionary(filenames, src=False, tgt=False):
        assert src ^ tgt
        return task.build_dictionary(
            filenames,
            tokenize_func=tokenizer.tokenize_list,
            workers=args['preprocess']['workers'],
            threshold=args['preprocess']['thresholdsrc'],
            nwords=args['preprocess']['nwordssrc'] if src else args['preprocess']['nwordstgt'],
            padding_factor=args['preprocess']['padding_factor'],
        )

    def build_vocab_dict(args):
        """Build vocabulary (dictionary) for source and target domain"""
        LOGGER.info('Build vocabularies...')
        # task = tasks.get_task(args['preprocess']['task'])
        src_dicts = OrderedDict()

        if args['preprocess']['joined_dictionary']:
            modalities = args['preprocess']['source_lang'] + [args['preprocess']['target_lang']]
            modalities = sorted(list(itertools.filterfalse(lambda modality: modality is None, modalities)))
            joined_dictionary_filename = os.path.join(args['preprocess']['destdir'],
                                                      '{}.dict.txt'.format('_'.join(modalities)))
            if os.path.exists(joined_dictionary_filename):
                LOGGER.info('Loading joint dict from {}'.format(joined_dictionary_filename))
                joined_dictionary = Dictionary.load_json(joined_dictionary_filename)
            else:
                joined_dictionary = build_dictionary(
                    [train_path(modality) for modality in modalities], src=True
                )
                LOGGER.info('Saving joint dict at {}'.format(joined_dictionary_filename))
                joined_dictionary.save_json(joined_dictionary_filename)

            for modality in modalities:
                src_dicts[modality] = joined_dictionary
            tgt_dict = joined_dictionary
        else:
            # src dict
            for modality in args['preprocess']['source_lang']:
                modality_dict_filename = os.path.join(args['preprocess']['destdir'], '{}.dict.json'.format(modality))
                if os.path.exists(modality_dict_filename):
                    LOGGER.info('Loading {} dict from {}'.format(modality, modality_dict_filename))
                    src_dicts[modality] = Dictionary.load_json(modality_dict_filename)
                else:
                    src_dicts[modality] = build_dictionary([train_path(modality)], src=True)
                    LOGGER.info('Saving {} dict at {}'.format(modality, modality_dict_filename))
                    src_dicts[modality].save_json(modality_dict_filename)
            # tgt dict
            if args['preprocess']['target_lang']:
                modality_dict_filename = os.path.join(args['preprocess']['destdir'],
                                                      '{}.dict.json'.format(args['preprocess']['target_lang']))
                if os.path.exists(modality_dict_filename):
                    LOGGER.info('Loading {} dict from {}'.format(modality, modality_dict_filename))
                    tgt_dict = Dictionary.load_json(modality_dict_filename)
                else:
                    tgt_dict = build_dictionary([train_path(args['preprocess']['target_lang'])], tgt=True)
                    LOGGER.info('Saving {} dict at {}'.format(modality, modality_dict_filename))
                    tgt_dict.save_json(modality_dict_filename)
            else:
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
        ds_file = '{}.bin'.format(output_file)
        ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(dict))
        merge_result(
            Binarizer.binarize(
                input_file, dict, lambda t: ds.add_item(t),
                tokenize=tokenizer.CSN_tokenizer(attr), offset=0, end=offsets[1]
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
                dict.unk_word,
            )
        )

    def make_dataset(vocab, input_prefix, output_prefix, lang, num_workers=1):
        if args['preprocess']['dataset_impl'] == "raw":
            in_file = file_name(input_prefix, lang)
            out_dir = args['preprocess']['destdir']
            os.makedirs(out_dir, exist_ok=True)
            logger.info('Copying {} into {}'.format(in_file, out_dir))
            shutil.copy(src=in_file, dst=args['preprocess']['destdir'])
        else:
            # TODO: please help me binarize it.
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
    args_ = Argues('preprocess.yml')  # train_sl
    LOGGER.info(args_)
    yaml_file = os.path.join(os.path.dirname(__file__), args_.yaml)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == "__main__":
    cli_main()
