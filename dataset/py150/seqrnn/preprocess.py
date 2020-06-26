#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""
import math
import os
import ujson
import itertools
from collections import namedtuple
# from ncc.utils.mp_ppool import PPool, cpu_count
from multiprocessing import Pool, cpu_count
from ncc.utils.mp_ppool import PPool

from ncc import tasks
from collections import (
    Counter,
)
from ncc.data import (
    indexed_dataset,
)
from ncc.data.tools.binarizer import Binarizer
from ncc.utils.util_file import load_yaml
from ncc import LOGGER
from ncc.utils import py150_utils
import json


def get_leaf_ids(types_):
    ids = {"leaf_ids": []}
    for i, v in enumerate(types_):
        if v is not None:
            ids["leaf_ids"].append(i)
    return ids


def get_value_ids(types_):
    ids = {"attr_ids": [], "num_ids": [], "name_ids": [], "param_ids": []}
    for i, v in enumerate(types_):
        if v == "attr":
            ids["attr_ids"].append(i)
        elif v == "Num":
            ids["num_ids"].append(i)
        elif v in {"NameStore", "NameLoad"}:
            ids["name_ids"].append(i)
        elif v == "NameParam":
            ids["param_ids"].append(i)
    return ids


def dataset_dest_prefix(args, output_prefix, lang):
    base = "{}/{}".format(args['preprocess']['destdir'], output_prefix)
    if lang is not None:
        lang_part = ".{}.{}".format(args['preprocess']['source_lang'], lang)
    elif args['preprocess']['only_source']:
        lang_part = ""
    else:
        lang_part = ".{}".format(args['preprocess']['source_lang'])

    return "{}{}".format(base, lang_part)


def dataset_dest_file(args, output_prefix, lang, extension):
    base = dataset_dest_prefix(args, output_prefix, lang)
    return "{}.{}".format(base, extension)


def string2dfs(line):
    line = json.loads(line)
    # ast = line[:args['preprocess']['n_ctx']]
    ast = line[:1000]
    assert len(ast) > 1
    ast = py150_utils.get_dfs(ast)
    return ast


def binarize(args, filename, vocab, output_prefix, lang, offset, end, append_eos=True):
    ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, lang, "bin"),
                                      impl=args['preprocess']['dataset_impl'], vocab_size=len(vocab))

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize(filename, vocab, consumer, tokenize=string2dfs, append_eos=append_eos,
                             offset=offset, end=end)
    ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
    return res


# TODO: Don't abstract it. Try to be consistent with Fairseq.
def main(args):
    LOGGER.info('mkdir for {} task'.format(args['preprocess']['task']))
    os.makedirs(args['preprocess']['destdir'], exist_ok=True)
    # 1. ***************build vocabulary***************
    # src_dicts, tgt_dict = build_vocab_dict(args, overwrite=True)
    task = tasks.get_task(args['preprocess']['task'])

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, lang):
        return os.path.join(args['preprocess']['destdir'], file_name(prefix, lang))

    def dict_path(lang):
        return dest_path("dict", lang) + ".txt"

    def build_dictionary(filenames, src=False, tgt=False):
        assert src ^ tgt
        return task.build_dictionary(
            filenames,
            workers=args['preprocess']['workers'],
            threshold=args['preprocess']['thresholdsrc'],
            nwords=args['preprocess']['nwordssrc'] if src else args['preprocess']['nwordstgt'],
            padding_factor=args['preprocess']['padding_factor'],
        )

    def train_path(lang):
        return "{}{}".format(args['preprocess']['trainpref'], ("." + lang) if lang else "")

    if args['preprocess']['srcdict']:
        src_dict = task.load_dictionary(args['preprocess']['srcdict'])
    else:
        assert args['preprocess']['trainpref'], "--trainpref must be set if --srcdict is not specified"
        dict_filename = dict_path(args['preprocess']['source_lang'])
        if os.path.exists(dict_filename):
            src_dict = task.load_dictionary(dict_filename)
        else:
            src_dict = build_dictionary([train_path(args['preprocess']['source_lang'])], src=True)
            #  save dictionary
            src_dict.save(dict_path(args['preprocess']['source_lang']))

    # 2. ***************build dataset********************
    def make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers):
        LOGGER.info("[{}] Dictionary: {} types".format(lang, len(vocab) - 1))
        n_seq_tok = [0, 0]
        replaced = Counter()

        def merge_result(worker_result):
            replaced.update(worker_result["replaced"])
            n_seq_tok[0] += worker_result["nseq"]
            n_seq_tok[1] += worker_result["ntok"]

        input_file = "{}{}".format(
            input_prefix, ("." + lang) if lang is not None else ""
        )
        lang = None
        offsets = Binarizer.find_offsets(input_file, num_workers)
        pool = None
        if num_workers > 1:
            pool = Pool(num_workers - 1)
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
                        offsets[worker_id + 1],
                    ),
                    callback=merge_result
                )
            pool.close()

        ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, lang, "bin"),
                                          impl=args['preprocess']['dataset_impl'], vocab_size=len(vocab))

        merge_result(
            Binarizer.binarize(
                input_file, vocab, lambda t: ds.add_item(t),
                tokenize=string2dfs,
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

    def make_dataset(vocab, input_prefix, output_prefix, lang, num_workers=1):
        if args['preprocess']['dataset_impl'] == "raw":
            thread_pool = PPool()
            _func = lambda line: py150_utils.separate_dps(json.loads(line), args['preprocess']['n_ctx'])
            MAX_SCRIPT_NUM = 50000  # avoid out of memory

            # load tok/type files
            with open(file_name(input_prefix, lang), 'r', encoding="utf-8") as tok_reader, \
                    open(file_name(input_prefix, 'type'), 'r', encoding="utf-8") as type_reader, \
                    open(dest_path(output_prefix, lang), 'w') as fout:
                def write_seqrnn_info(tokens, types):
                    for (tokens, ext), (types_, _) in zip(tokens, types):
                        if len(tokens) > 1:
                            if args['preprocess']['id_type'] == "leaf":
                                json.dump(get_leaf_ids(types_), fp=fout)
                            elif args['preprocess']['id_type'] == "value":
                                json.dump(get_value_ids(types_), fp=fout)
                            elif args['preprocess']['id_type'] == "all":
                                ids = get_leaf_ids(types_)
                                ids.update(get_value_ids(types_))
                                json.dump(ids, fp=fout)
                            else:
                                json.dump([tokens, ext], fp=fout)
                            fout.write("\n")

                params = []
                for tokens, types in zip(tok_reader, type_reader):
                    params.append((tokens, types,))
                    if len(params) >= MAX_SCRIPT_NUM:
                        params = thread_pool.feed(lambda *args: list(map(_func, args)), params)
                        for tokens, types in params:
                            write_seqrnn_info(tokens, types)
                        del params
                        params = []
                if len(params) > 0:
                    params = thread_pool.feed(lambda *args: list(map(_func, args)), params)
                    for tokens, types in params:
                        write_seqrnn_info(tokens, types)
                    del params

                # token_list, type_list, count = [None] * MAX_SCRIPT_NUM, [None] * MAX_SCRIPT_NUM, 0
                # for tokens, types in zip(tok_reader, type_reader):
                #     token_list[count], type_list[count], count = tokens, types, count + 1
                #     if count >= MAX_SCRIPT_NUM:
                #         token_list = thread_pool.feed(_func, token_list, one_params=True)
                #         type_list = thread_pool.feed(_func, type_list, one_params=True)
                #         exit()
                #         for tokens, types in zip(token_list, type_list):
                #             write_seqrnn_info(tokens, types)
                #         token_list, type_list, count = [None] * MAX_SCRIPT_NUM, [None] * MAX_SCRIPT_NUM, 0
                # if count > 0:
                #     token_list = thread_pool.feed(_func, token_list, one_params=True)
                #     type_list = thread_pool.feed(_func, type_list, one_params=True)
                #     for tokens, types in zip(token_list, type_list):
                #         write_seqrnn_info(tokens, types)

        else:
            # TODO: please help me binarize it.
            make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers)

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

    # build_dataset(args, src_dicts, tgt_dict)
    make_all(args['preprocess']['source_lang'], src_dict)

    # # 3. ***************generate ids********************
    # def generate_ids(input_prefix, output_prefix, lang):
    #     with open(file_name(input_prefix, lang), "r", encoding="utf-8") as f, \
    #             open(dest_path(output_prefix, lang), "w") as fout:
    #         for line in f.readlines():
    #             dp = json.loads(line.strip())
    #             # asts = separate_dps(dp, args.n_ctx)
    #             asts = py150_utils.separate_dps(dp, args['preprocess']['n_ctx'])
    #
    #             for ast, _ in asts:
    #                 ids = {}
    #                 if len(ast) > 1:
    #                     if args['preprocess']['id_type'] in {"leaf", "all"}:
    #                         ids.update(py150_utils.get_leaf_ids(ast))
    #                     if args['preprocess']['id_type'] in {"value", "all"}:
    #                         ids.update(py150_utils.get_value_ids(ast))
    #                     if args['preprocess']['id_type'] in {"type", "all"}:
    #                         ids.update(py150_utils.get_type_ids(ast))
    #
    #                     json.dump(ids, fp=fout)
    #                     fout.write("\n")
    #
    # generate_ids(args['preprocess']['trainpref'], "train.generate_id", args['preprocess']['source_lang'])


def cli_main():
    """
    seq rnn data generation:

    Examples:
        def add(a, b):\n  return a + b

    1） parse code into ast with DFS
    2) collect code/type token from ast with DFS for dictionary generation,
        and parse a ast into smaller sub-asts, because some ast are too big
    3) extract info from raw ast:
        --token: ['def', 'add', '(', 'a', ',', 'b', ')', ':', 'return', '(', 'a', '+', 'b', ')'], 0 (start index of raw ast,
                    for calculating loss)
        --leaf: {'leaf_ids': [1, 3, 5, 10, 12]}
        --value: {'attr_ids': [], 'num_ids': [], 'name_ids': [10, 12], 'param_ids': []}
        --all: {'leaf_ids': [1, 3, 5, 10, 12], 'attr_ids': [], 'num_ids': [], 'name_ids': [10, 12], 'param_ids': []}
    """
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
