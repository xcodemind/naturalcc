#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""
from collections import Counter
from collections import OrderedDict
from multiprocessing import Pool
import shutil
from ncc.data import indexed_dataset
from ncc.data.binarizer import Binarizer
import os
import sys
from collections import namedtuple
import torch
from ncc import tasks
from ncc.utils.util_file import load_yaml
from ncc.data import Dictionary
from ncc.utils import (
    utils, tokenizer
)
from ncc import LOGGER
from typing import Dict, List
import ujson
import re
S_SEP = '<S_SEP>'
CLS = '<CLS>'
# S_MASK = '<MASK>'
# from dataset.codesearchnet.summarization.preprocess_helper import insert_sep_token

# try:
#     from .preprocess_helper import insert_sep_tokens
# except:
#     from dataset.codesearchnet.summarization.preprocess_helper import insert_sep_tokens

# train_path is for build dictionary. Currently, we only insert <S_SEP> to code modality
def train_path(args, lang):
    # if lang in args['preprocess']['source_lang']:
    # if args['preprocess']['inserted']:
    #     if lang == 'code':
    #         return "{}_inserted{}".format(args['preprocess']['trainpref'], ("." + lang) if lang else "")
    #     else:
    #         return "{}{}".format(args['preprocess']['trainpref'], ("." + lang) if lang else "")
    # else:
    return "{}{}".format(args['preprocess']['trainpref'], ("." + lang) if lang else "")


def file_name(prefix, lang, inserted=False):
    fname = prefix  # + '_inserted' if inserted else prefix
    if lang is not None:
        fname += ".{lang}".format(lang=lang)
    return fname


######################################################################
# dictionary functions
######################################################################
def dict_path(args, lang):
    return dest_path(args, "dict", lang) + ".txt"


def build_dictionary(args, task, filenames, src=False, tgt=False):
    assert src ^ tgt
    return task.build_dictionary(
        filenames,
        # modality=modality,
        tokenize_func=tokenizer.CSN_tokenizer(args['preprocess']['source_lang']),
        workers=args['preprocess']['workers'],
        threshold=args['preprocess']['thresholdsrc'] if src else args['preprocess']['thresholdtgt'],
        nwords=args['preprocess']['nwordssrc'] if src else args['preprocess']['nwordstgt'],
        padding_factor=args['preprocess']['padding_factor'],
    )

# 这个名字没取好，经常很难理解，而且save操作也不应该放在这里，所以就不抽象出来了
# def load_dict(args: Dict, task, modality: str, overwrite: bool):
#     """load dict from (default) dictionary file path. if not exit, load from raw data and save it at default path"""
#     dict_filenames = dict_path(args, modality)
#     all_file_exit = all([os.path.exists(filename) for filename in dict_filenames])
#     if all_file_exit and (not overwrite):
#         LOGGER.info('Dict({}) exists and overwrite=False, skip this.'.format(dict_filenames))
#         # dicts = [
#         #     Dictionary.load(filename)
#         #     for filename in dict_filenames
#         # ]
#         dicts = task.load_dictionaries(dict_filenames)
#         if len(dicts) == 1:
#             dicts = dicts[0]
#     else:
#         # update dict from data
#         dicts = build_dictionary(args, task, modality, [train_path(args, modality)], src=True)
#
#     return dicts


def build_vocab_dict(args: Dict, overwrite: bool = False):
    """Build vocabulary (dictionary) for source and target domain"""
    LOGGER.info('Build vocabularies...')
    task = tasks.get_task(args['preprocess']['task'])
    # src_dicts = OrderedDict()
    assert args['preprocess']['trainpref'], RuntimeError('Build vocabularies from train dataset, but it is null.')
    target = not args['preprocess']['only_source']

    # for joined dictionary
    if args['preprocess']['joined_dictionary']:
        # for modality in args['preprocess']['source_lang']:
        # src_dicts[modality] = load_dict(args, task, modality, overwrite)
        # dict_filenames = dict_path(args, modality)
        # all_file_exit = all([os.path.exists(filename) for filename in dict_filenames])
        # if all_file_exit and (not overwrite):
        # dict_filename = dict_path(args, args['preprocess']['source_lang'])
        # if dict_filename and (not overwrite):
        #     LOGGER.info('Dict({}) exists and overwrite=False, skip this.'.format(dict_filename))
        if args['preprocess']['srcdict']:
            src_dict = task.load_dictionary(args['preprocess']['srcdict'])
            # if len(dicts) == 1:
            #     dicts = dicts[0]
        else:
            # build dictionary from source_lang and target_lang
            src_dict = build_dictionary(args, task, {train_path(args, lang) for lang in [args['preprocess']['source_lang'], args['preprocess']['target_lang']]}, src=True)

        # the tgt_dict equals
        tgt_dict = src_dict
        # tgt_dict = src_dicts.values()[-1]

    else:
        # for modality in args['preprocess']['source_lang']:
        # src_dicts[modality] = load_dict(args, task, modality, overwrite)
        # dict_filename = dict_path(args, args['preprocess']['source_lang'])
        # all_file_exit = all([os.path.exists(filename) for filename in dict_filenames])
        # if all_file_exit and (not overwrite):
        # if dict_filename and (not overwrite):
        #     LOGGER.info('Dict({}) exists and overwrite=False, skip this.'.format(dict_filename))
        if args['preprocess']['srcdict']:
            src_dict = task.load_dictionary(args['preprocess']['srcdict'])
            # if len(dicts) == 1:
            #     dicts = dicts[0]
        else:
            # update dict from data
            src_dict = build_dictionary(args, task, train_path(args, args['preprocess']['source_lang']), src=True)

        if target:
            # tgt_dict = load_dict(args, task, args['preprocess']['target_lang'], overwrite)
            tgt_dict = build_dictionary(args, task, train_path(args, args['preprocess']['target_lang']), src=True)
        else:
            tgt_dict = None

    # save src dict
    # for modality, src_dict in src_dicts.items():
    #     dict_filename = dict_path(args, modality)
    #     LOGGER.info('Save dict(s) for {} in {}.'.format(modality, dict_filename))
    #     # if isinstance(dicts, Dictionary) and len(dict_filenames) == 1:
    #     #     dicts.save(dict_filenames[0])
    #     # else:
    #     #     for dict_filename, dict in zip(dict_filenames, dicts):
    #     #         dict.save(dict_filename)
    #     src_dict.save(dict_filename)
    src_dict.save(dict_path(args, args['preprocess']['source_lang']))
    # save tgt dict
    tgt_dict.save(dict_path(args, args['preprocess']['target_lang']))

    return src_dict, tgt_dict


######################################################################
# dataset functions
######################################################################

def dest_path(args, prefix, lang):
    return os.path.join(args['preprocess']['destdir'], file_name(prefix, lang))


def dataset_dest_prefix(args, output_prefix, lang):
    # base = "{}/{}".format(args['preprocess']['destdir'], output_prefix)
    # if lang is not None:
    #     lang_part = ".{}-{}.{}".format(args['preprocess']['source_lang'], args['preprocess']['target_lang'], lang)
    # elif args['preprocess']['only_source']:
    #     lang_part = ""
    # else:
    #     lang_part = ".{}-{}".format(args['preprocess']['source_lang'], args['preprocess']['target_lang'])
    #
    # return "{}{}".format(base, lang_part)
    """dataset file name without extension"""
    dest_file = os.path.join(args['preprocess']['destdir'], '.'.join([output_prefix, lang]))
    return dest_file


def dataset_dest_file(args, output_prefix, lang, extension):
    base = dataset_dest_prefix(args, output_prefix, lang)
    return "{}.{}".format(base, extension)


def binarize(args, filename, vocab, output_prefix, lang, offset, end, append_eos=True):
    ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, lang, "bin"),
                                      impl=args['preprocess']['dataset_impl'], vocab_size=len(vocab))

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize(filename, vocab, consumer, append_eos=append_eos,
                             offset=offset, end=end)
    ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
    return res


def get_offsets(input_file, num_workers):
    return Binarizer.find_offsets(input_file, num_workers)


def make_binary_dataset(args, vocab, input_prefix, output_prefix, lang, num_workers, inserted=False):
    LOGGER.info("[{}] Dictionary: {} types".format(lang, len(vocab) - 1))
    n_seq_tok = [0, 0]
    replaced = Counter()

    def merge_result(worker_result):
        replaced.update(worker_result["replaced"])
        n_seq_tok[0] += worker_result["nseq"]
        n_seq_tok[1] += worker_result["ntok"]
    # input_prefix = input_prefix + '_inserted' if inserted else input_prefix
    input_file = "{}{}".format(
        input_prefix, ("." + lang) if lang is not None else ""
    )
    # input_file = os.path.join(input_prefix, 'code', 'train', 'train_all.txt', )
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

    ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, lang, "bin"),
                                      impl=args['preprocess']['dataset_impl'], vocab_size=len(vocab))
    merge_result(
        Binarizer.binarize(
            input_file, vocab, lambda t: ds.add_item(t),
            offset=0, end=offsets[1]
        )
    )
    if num_workers > 1:
        pool.join()
        for worker_id in range(1, num_workers):
            prefix = "{}{}".format(output_prefix, worker_id)
            temp_file_path = dataset_dest_prefix(args, prefix, lang)
            ds.merge_file_(temp_file_path)
            os.remove(indexed_dataset.data_file_path(temp_file_path))
            os.remove(indexed_dataset.index_file_path(temp_file_path))

    ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))

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
    if args['preprocess']['dataset_impl'] == "raw":
        # Copy original text file to destination folder
        output_text_file = dest_path(args,
                                     output_prefix,
                                     # + ".{}-{}".format(args['preprocess']['source_lang'], args['preprocess']['target_lang'])
                                     lang,
                                     )
        if lang == 'docstring':  # since docstring did't be inserted <S_SEP>, therefore the inserted should be set to False
            shutil.copyfile(file_name(input_prefix, lang), output_text_file)
        else:
            shutil.copyfile(file_name(input_prefix, lang), output_text_file)
    else:
        # if lang == 'docstring':
        #     make_binary_dataset(args, vocab, input_prefix, output_prefix, lang, num_workers)
        # else:
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


def build_dataset(args: Dict, src_dict: Dictionary, tgt_dict: Dictionary):
    """build dataset for modal"""
    # for modality, src_dict in src_dict.items():
    # LOGGER.info('Building dataset for {}'.format(modality))
    # if modality == 'path':
    #     make_all(args, modality, src_dict)
    # elif modality == 'code':
    #     make_all(args, modality, src_dict)
    # else:
    make_all(args, args['preprocess']['source_lang'], src_dict)
    target = not args['preprocess']['only_source']
    if target:
        make_all(args, args['preprocess']['target_lang'], tgt_dict)




def main(args):
    LOGGER.info('mkdir for {} task'.format(args['preprocess']['task']))
    os.makedirs(args['preprocess']['destdir'], exist_ok=True)
    # 0. insert special tokens for the code modality
    # if args['preprocess']['inserted']:
    #     insert_sep_token(args, args['preprocess']['trainpref'], 'train_inserted', 'code')
    #     insert_sep_token(args, args['preprocess']['validpref'], 'valid_inserted', 'code')
    #     insert_sep_token(args, args['preprocess']['testpref'], 'test_inserted', 'code')

    # 1. build vocabulary
    src_dict, tgt_dict = build_vocab_dict(args, overwrite=True)
    # 2. build dataset
    build_dataset(args, src_dict, tgt_dict)


def cli_main():
    Argues = namedtuple('Argues', 'yaml')
    args_ = Argues('preprocess_hicodebert_sm.yml')  # train_sl
    LOGGER.info(args_)
    yaml_file = os.path.join(os.path.dirname(__file__), args_.yaml)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == "__main__":
    cli_main()
