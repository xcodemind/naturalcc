#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""
from typing import *

from collections import Counter
from collections import OrderedDict
from itertools import zip_longest
from multiprocessing import Pool
import shutil
from ncc.data import indexed_dataset
from ncc.data.binarizer import Binarizer
import os
import sys
from collections import namedtuple
import torch
from ncc import tasks
from ncc.data import Dictionary
from ncc.utils.util_file import load_yaml
from ncc.utils import (
    utils, tokenizer
)
from ncc import LOGGER


def train_path(args, lang):
    """get train data path"""
    return "{}{}".format(args['preprocess']['trainpref'], ("." + lang) if lang else "")


def file_name(prefix, lang):
    fname = prefix
    if lang is not None:
        fname += ".{lang}".format(lang=lang)
    return fname


######################################################################
# dictionary functions
######################################################################

def dict_path(args: Dict, modality: str) -> List[str]:
    """Get vocab token file. This file is not dictionary file. Dictionary file will be written later."""

    def _default_path():
        if modality == 'path':
            dict_path = [
                os.path.join(os.path.join(args['preprocess']['destdir'], 'dict.{}_border.txt'.format(modality))),
                os.path.join(os.path.join(args['preprocess']['destdir'], 'dict.{}_center.txt'.format(modality))),
            ]
        else:
            dict_path = [
                os.path.join(os.path.join(args['preprocess']['destdir'], 'dict.{}.txt'.format(modality)))
            ]
        return dict_path

    if '{}_dict'.format(modality) in args['preprocess']:
        dict_path = args['preprocess']['{}_dict'.format(modality)]
        if dict_path is None:
            return _default_path()
        else:
            return dict_path
    else:
        return _default_path()


def build_dictionary(args: Dict, task, modality: str, filenames, src=False, tgt=False) \
        -> Union[List[Dictionary], Dictionary]:
    """build dictionary for modality"""
    assert src ^ tgt, RuntimeError('Cannot build dictionary for source and target domain at the same time.')
    return task.build_dictionary(
        filenames,
        modality=modality,
        tokenize_func=tokenizer.CSN_tokinzer(modality),
        workers=args['preprocess']['workers'],
        threshold=args['preprocess']['thresholdsrc'] if src else args['preprocess']['thresholdtgt'],
        nwords=args['preprocess']['nwordssrc'] if src else args['preprocess']['nwordstgt'],
        padding_factor=args['preprocess']['padding_factor'],
    )


def load_dict(args: Dict, task, modality: str, overwrite: bool):
    """load dict from (default) dictionary file path. if not exit, load from raw data and save it at default path"""
    dict_filenames = dict_path(args, modality)
    all_file_exit = all([os.path.exists(filename) for filename in dict_filenames])
    if all_file_exit and (not overwrite):
        LOGGER.info('Dict({}) exists and overwrite=False, skip this.'.format(dict_filenames))
        dicts = [
            Dictionary.load(filename)
            for filename in dict_filenames
        ]
        if len(dicts) == 1:
            dicts = dicts[0]
    else:
        # update dict from data
        dicts = build_dictionary(args, task, modality, [train_path(args, modality)], src=True)
        # save dict
        LOGGER.info('Save dict(s) for {} in {}.'.format(modality, dict_filenames))
        if isinstance(dicts, Dictionary) and len(dict_filenames) == 1:
            dicts.save(dict_filenames[0])
        else:
            for dict_filename, dict in zip(dict_filenames, dicts):
                dict.save(dict_filename)
    return dicts


def build_vocab_dict(args: Dict, overwrite: bool = False):
    """Build vocabulary (dictionary) for source and target domain"""
    LOGGER.info('Build vocabularies...')
    task = tasks.get_task(args['preprocess']['task'])
    src_dicts = OrderedDict()
    assert args['preprocess']['trainpref'], RuntimeError('Build vocabularies from train dataset, but it is null.')
    for modality in args['preprocess']['source_lang']:
        src_dicts[modality] = load_dict(args, task, modality, overwrite)
    return src_dicts


######################################################################
# dataset functions
######################################################################


def dest_path(args, prefix, lang):
    return os.path.join(args['preprocess']['destdir'], file_name(prefix, lang))


def dataset_dest_prefix(args, output_prefix, lang):
    """dataset file name without extension"""
    dest_file = os.path.join(args['preprocess']['destdir'], '.'.join([output_prefix, lang]))
    return dest_file


def dataset_dest_file(args, output_prefix, lang, extension: str):
    """generate bin file for each modality"""
    base = dataset_dest_prefix(args, output_prefix, lang)
    return "{}.{}".format(base, extension)


def binarize(args, filename, vocab, output_prefix, lang, offset, end, append_eos=True):
    """binarize function for multi-processing"""
    ds_file = dataset_dest_file(args, output_prefix, lang, extension='bin')
    ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(vocab))

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize(filename, vocab, consumer, tokenize=tokenizer.tokenize_list,
                             append_eos=append_eos, offset=offset, end=end)
    ds.finalize(dataset_dest_file(args, output_prefix, lang, extension="idx"))
    return res


def get_offsets(input_file, num_workers):
    """get pointers' start index for multi-processing"""
    return Binarizer.find_offsets(input_file, num_workers)


def make_binary_dataset(args, vocab, input_prefix, output_prefix, lang, num_workers):
    """make binary dataset"""
    LOGGER.info("[{}] Dictionary: {} types".format(lang, len(vocab) - 1))
    n_seq_tok = [0, 0]
    replaced = Counter()  # save un-recorded tokens

    def merge_result(worker_result):
        replaced.update(worker_result["replaced"])
        n_seq_tok[0] += worker_result["nseq"]
        n_seq_tok[1] += worker_result["ntok"]

    input_file = "{}{}".format(
        input_prefix, ("." + lang) if lang is not None else ""
    )
    # split a file into different parts
    # if use multi-processing, we first process 2nd to last file
    offsets = Binarizer.find_offsets(input_file, num_workers)
    pool = None
    if num_workers > 1:
        pool = Pool(processes=num_workers - 1)
        for worker_id in range(1, num_workers):
            prefix = "{}{}".format(output_prefix, worker_id)
            pool.apply_async(
                binarize,
                (
                    args,
                    input_file,
                    vocab,
                    prefix,
                    lang,
                    offsets[worker_id],
                    offsets[worker_id + 1]
                ),
                callback=merge_result
            )
        pool.close()
    # process 1th file, if multi-processing available. If not, process all file
    ds_file = dataset_dest_file(args, output_prefix, lang, extension='bin')
    ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(vocab))
    merge_result(
        Binarizer.binarize(
            input_file, vocab, lambda t: ds.add_item(t),
            tokenize=tokenizer.tokenize_list, offset=0, end=offsets[1]
        )
    )
    if num_workers > 1:
        pool.join()
        # merge sub-processors' index and data files into final files and delete them.
        for worker_id in range(1, num_workers):
            prefix = "{}{}".format(output_prefix, worker_id)
            temp_file_path = dataset_dest_prefix(args, prefix, lang)
            ds.merge_file_(temp_file_path)
            os.remove(indexed_dataset.data_file_path(temp_file_path))
            os.remove(indexed_dataset.index_file_path(temp_file_path))

    ds.finalize(dataset_dest_file(args, output_prefix, lang, extension="idx"))

    LOGGER.info(
        "[{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
            lang,
            input_file,
            n_seq_tok[0],
            n_seq_tok[1],
            100 * sum(replaced.values()) / n_seq_tok[1],
            vocab.unk_word,
        )
    )


def make_dataset(args, vocab, input_prefix, output_prefix, lang, num_workers=1):
    """
    build raw/bin dataset
    1) raw dataset: copy raw files
    2) bin dataset: build bin
    """
    if args['preprocess']['dataset_impl'] == "raw":
        # Copy original text file to destination folder
        output_text_file = dest_path(args,
                                     output_prefix,
                                     # + ".{}-{}".format(args['preprocess']['source_lang'], args['preprocess']['target_lang'])
                                     lang,
                                     )
        shutil.copyfile(file_name(input_prefix, lang), output_text_file)
    else:
        make_binary_dataset(args, vocab, input_prefix, output_prefix, lang, num_workers)


def make_all(args, lang, vocab):
    if args['preprocess']['trainpref']:
        make_dataset(args, vocab, args['preprocess']['trainpref'], "train", lang,
                     num_workers=args['preprocess']['workers'])
    if args['preprocess']['validpref']:
        for k, validpref in enumerate(args['preprocess']['validpref'].split(",")):
            outprefix = "valid{}".format(k) if k > 0 else "valid"
            make_dataset(args, vocab, validpref, outprefix, lang, num_workers=args['preprocess']['workers'])
    if args['preprocess']['testpref']:
        for k, testpref in enumerate(args['preprocess']['testpref'].split(",")):
            outprefix = "test{}".format(k) if k > 0 else "test"
            make_dataset(args, vocab, testpref, outprefix, lang, num_workers=args['preprocess']['workers'])


def build_dataset(args: Dict, dicts: Dict[str, Dictionary]):
    """build dataset for modal"""
    for modal, dict in dicts.items():
        LOGGER.info('Building dataset for {}'.format(modal))
        if modal == 'path':
            make_all(args, modal, dict)


def main(args):
    LOGGER.info('mkdir for {} task'.format(args['preprocess']['task']))
    os.makedirs(args['preprocess']['destdir'], exist_ok=True)
    src_dicts = build_vocab_dict(args, overwrite=False)
    build_dataset(args, src_dicts)


def cli_main():
    Argues = namedtuple('Argues', 'yaml')
    args_ = Argues('preprocess_summarization.yml')  # train_sl
    LOGGER.info(args_)
    yaml_file = os.path.join(os.path.dirname(__file__), args_.yaml)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == "__main__":
    cli_main()
    sys.exit()
