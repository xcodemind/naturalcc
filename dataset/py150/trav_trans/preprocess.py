#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""
import os
from collections import namedtuple
from multiprocessing import Pool

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


# TODO: Don't abstract it. Try to be consistent with Frirseq.
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
                                          impl=args.dataset_impl, vocab_size=len(vocab))
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

    def make_dataset(vocab, input_prefix, output_prefix, lang, num_workers=1):
        if args['preprocess']['dataset_impl'] == "raw":
            # Copy original text file to destination folder
            # output_text_file = dest_path(
            #     output_prefix, # '.bpe' #".{}-{}".format(args.source_lang, args.target_lang),
            #     lang,
            # ))
            # shutil.copyfile(file_name(input_prefix, lang), output_text_file)
        #     TODO: parse json to txt file, one line one traversal, please help me parallize it.
            with open(file_name(input_prefix, lang), 'r', encoding="utf-8") as f, \
                    open(dest_path(output_prefix, lang), 'w') as fout:
                for line in f.readlines():
                    line = json.loads(line)
                    asts = py150_utils.separate_dps(line, args['preprocess']['n_ctx'])
                    for ast, extended in asts:
                        if len(ast) > 1:
                            json.dump(py150_utils.get_dfs(ast), fp=fout)
                            fout.write('\n')
        else:
            # TODO: please help me binarize it.
            make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers)

    def make_all(lang, vocab):
        if args['preprocess']['trainpref']:
            make_dataset(vocab, args['preprocess']['trainpref'], "train", lang, num_workers=args['preprocess']['workers'])
        if args['preprocess']['validpref']:
            for k, validpref in enumerate(args['preprocess']['validpref'].split(",")):
                outprefix = "valid{}".format(k) if k > 0 else "valid"
                make_dataset(vocab, validpref, outprefix, lang, num_workers=args['preprocess']['workers'])
        if args['preprocess']['testpref']:
            for k, testpref in enumerate(args['preprocess']['testpref'].split(",")):
                outprefix = "test{}".format(k) if k > 0 else "test"
                make_dataset(vocab, testpref, outprefix, lang, num_workers=args['preprocess']['workers'])

    def binarize(args, filename, vocab, output_prefix, lang, offset, end, append_eos=True):
        ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, lang, "bin"),
                                          impl=args['preprocess']['dataset_impl'], vocab_size=len(vocab))

        def consumer(tensor):
            ds.add_item(tensor)

        res = Binarizer.binarize(filename, vocab, consumer, append_eos=append_eos,
                                 offset=offset, end=end)
        ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
        return res

    def dataset_dest_prefix(args, output_prefix, lang):
        base = "{}/{}".format(args['preprocess']['destdir'], output_prefix)
        if lang is not None:
            lang_part = ".{}-{}.{}".format(args['preprocess']['source_lang'], args['preprocess']['target_lang'], lang)
        elif args['preprocess']['only_source']:
            lang_part = ""
        else:
            lang_part = ".{}-{}".format(args['preprocess']['source_lang'], args['preprocess']['target_lang'])

        return "{}{}".format(base, lang_part)

    def dataset_dest_file(args, output_prefix, lang, extension):
        base = dataset_dest_prefix(args, output_prefix, lang)
        return "{}.{}".format(base, extension)

    # build_dataset(args, src_dicts, tgt_dict)
    make_all(args['preprocess']['source_lang'], src_dict)

    # 3. ***************generate ids********************
    def generate_ids(input_prefix, output_prefix, lang):
        with open(file_name(input_prefix, lang), "r", encoding="utf-8") as f, \
                open(dest_path(output_prefix, lang), "w") as fout:
            for line in f.readlines():
                dp = json.loads(line.strip())
                # asts = separate_dps(dp, args.n_ctx)
                asts = py150_utils.separate_dps(dp, args['preprocess']['n_ctx'])

                for ast, _ in asts:
                    ids = {}
                    if len(ast) > 1:
                        if args['preprocess']['id_type'] in {"leaf", "all"}:
                            ids.update(py150_utils.get_leaf_ids(ast))
                        if args['preprocess']['id_type'] in {"value", "all"}:
                            ids.update(py150_utils.get_value_ids(ast))
                        if args['preprocess']['id_type'] in {"type", "all"}:
                            ids.update(py150_utils.get_type_ids(ast))

                        json.dump(ids, fp=fout)
                        fout.write("\n")

    generate_ids(args['preprocess']['trainpref'], "train.generate_id", args['preprocess']['source_lang'])


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
