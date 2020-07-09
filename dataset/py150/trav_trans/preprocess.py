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

MAX_BATCH_SIZE = 5000


def tokenize_func(line):
    dp = []
    for node in json.loads(line):
        if "value" in node:
            dp.append(node["value"])
        else:
            dp.append(node["type"])
    return dp


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
            tokenize_func=tokenize_func,
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
        dict_filename = dest_path("dict", args['preprocess']['source_lang']) + ".json"
        if os.path.exists(dict_filename):
            LOGGER.info('Loading dict from {}'.format(dict_filename))
            src_dict = task.load_dictionary(dict_filename)
        else:
            src_dict = build_dictionary([train_path(args['preprocess']['source_lang'])], src=True)
            #  save dictionary
            LOGGER.info('Saving dict from {}'.format(dict_filename))
            src_dict.save_json(dict_filename)

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
            # TODO: parse json to txt file, one line one traversal, please help me parallize it.
            """
            because only 1 thread is allowed to write file, we have to use multi-processing for deal with data
            and merge results from CPUs into a block and then dumps such block. 
            """

            def _func(line):
                line = py150_utils.separate_dps(json.loads(line.strip()), args['preprocess']['n_ctx'])
                line = [py150_utils.get_dfs(ast) + [ext] for ast, ext in line if len(ast) > 1]
                # line = [json.dumps([py150_utils.get_dfs(ast), ext]) for ast, ext in line if len(ast) > 1]
                return line

            with PPool() as thread_pool:
                with open(file_name(input_prefix, lang), 'r', encoding="utf-8") as f, \
                        open(dest_path(output_prefix, lang), 'w') as fout:
                    def _write(result):
                        for res in itertools.chain(*result):
                            print(json.dumps(res), file=fout)

                    batch_data = []
                    for line in f:
                        batch_data.append(line)
                        if len(batch_data) >= MAX_BATCH_SIZE:
                            result = thread_pool.feed(_func, batch_data, one_params=True)
                            _write(result)
                            del batch_data
                            batch_data = []

                    if len(batch_data) > 0:
                        result = thread_pool.feed(_func, batch_data, one_params=True)
                        _write(result)
                        del batch_data


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

    make_all(args['preprocess']['source_lang'], src_dict)

    # 3. ***************generate ids********************
    def generate_ids(input_prefix, output_prefix, lang):
        def _func(line):
            line = py150_utils.separate_dps(json.loads(line.strip()), args['preprocess']['n_ctx'])
            tmp = []
            for ast, _ in line:
                if len(ast) > 1:
                    ids = {}
                    if args['preprocess']['id_type'] in {"leaf", "all"}:
                        ids.update(py150_utils.get_leaf_ids(ast))
                    if args['preprocess']['id_type'] in {"value", "all"}:
                        ids.update(py150_utils.get_value_ids(ast))
                    if args['preprocess']['id_type'] in {"type", "all"}:
                        ids.update(py150_utils.get_type_ids(ast))
                    tmp.append(ids)
            return tmp

        with PPool() as thread_pool:
            with open(file_name(input_prefix, lang), "r", encoding="utf-8") as f, \
                    open(dest_path(output_prefix, 'ids'), "w") as fout:
                def _write(result):
                    for res in itertools.chain(*result):
                        print(json.dumps(res), file=fout)

                batch_data = []
                for line in f:
                    batch_data.append(line)
                    if len(batch_data) >= MAX_BATCH_SIZE:
                        result = thread_pool.feed(_func, batch_data, one_params=True)
                        _write(result)
                        del batch_data
                        batch_data = []

                if len(batch_data) > 0:
                    result = thread_pool.feed(_func, batch_data, one_params=True)
                    _write(result)
                    del batch_data

    def make_all_ids():
        if args['preprocess']['trainpref']:
            generate_ids(args['preprocess']['trainpref'], "train", args['preprocess']['source_lang'])
        if args['preprocess']['validpref']:
            generate_ids(args['preprocess']['validpref'], "valid", args['preprocess']['source_lang'])
        if args['preprocess']['testpref']:
            generate_ids(args['preprocess']['testpref'], "test", args['preprocess']['source_lang'])

    make_all_ids()


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
