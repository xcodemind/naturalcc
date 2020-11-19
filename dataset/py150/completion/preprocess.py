#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""
from typing import Dict
import os
import ujson
import shutil
from collections import namedtuple
from multiprocessing import Pool
from ncc.utils.util_graph import (build_graph, tree2dgl)
from ncc import tasks
from collections import Counter
from ncc.data import (
    indexed_dataset,
)
from ncc.data.completion.completion_dictionary import CompletionDictionary as Dictionary
from ncc.data.completion.completion_binarizer import CompletionBinarizer as Binarizer
from ncc.utils import tokenizer
from ncc.utils.util_file import load_yaml
from ncc import LOGGER
from collections import OrderedDict


def separate_list(ast, max_len):
    """
    Handles training / evaluation on long ASTs by splitting
    them into smaller ASTs of length max_len, with a sliding
    window of max_len / 2.

    Example: for an AST ast with length 1700, and max_len = 1000,
    the output will be:
    [[ast[0:1000], 0], [ast[500:1500], 1000], [ast[700:1700], 1500]]

    Input:
        ast : List[Dictionary]
            List of nodes in pre-order traversal.
        max_len : int

    Output:
        aug_asts : List[List[List, int]]
            List of (ast, beginning idx of unseen nodes)
    """
    half_len = int(max_len / 2)
    if len(ast) <= max_len:
        return [[ast, 0]]

    aug_asts = [[ast[:max_len], 0]]
    i = half_len
    while i < len(ast) - max_len:
        aug_asts.append([ast[i: i + max_len], half_len])
        i += half_len
    idx = max_len - (len(ast) - (i + half_len))
    aug_asts.append([ast[-max_len:], idx])
    return aug_asts


def get_code_tokens(line):
    """get DFS leaf node of ast"""
    ast = ujson.loads(line)
    code_tokens = [node["value"] for node in ast if "value" in node]
    return code_tokens


def binarize(args: Dict, filename: str, dict: Dictionary, in_file: str,
             offset: int, end: int, append_eos: bool = False):
    """binarize function for multi-processing"""
    ds_file = '{}.mmap'.format(in_file)
    ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(dict))

    ext_ds_file = '{}.ext.mmap'.format(in_file)
    ext_ds = indexed_dataset.make_builder(ext_ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(dict))

    def consumer(data, start_idx):
        ds.add_item(data)
        ext_ds.add_item(start_idx)

    def seperate_tokenize(line):
        tokens = get_code_tokens(line)
        tokens = separate_list(tokens, args['preprocess']['max_len'])
        return tokens

    res = Binarizer.binarize_seperate(filename, dict, consumer, tokenize=seperate_tokenize,
                                      append_eos=append_eos, offset=offset, end=end)
    ds.finalize('{}.idx'.format(in_file))
    ext_ds.finalize('{}.ext.idx'.format(in_file))
    return res


def main(args):
    task = tasks.get_task(args['preprocess']['task'])
    LOGGER.info('mkdir {} for {} task'.format(args['preprocess']['destdir'], args['preprocess']['task']))
    os.makedirs(args['preprocess']['destdir'], exist_ok=True)

    def train_path():
        return "{}.json".format(args['preprocess']['trainpref'])

    def file_name(prefix, lang):
        return f"{prefix}.{lang}"

    def dest_path(prefix, lang):
        return os.path.join(args['preprocess']['destdir'], file_name(prefix, lang))

    def dict_path(lang):
        return dest_path(lang, "dict") + ".json"

    # if not args['preprocess']['srcdict'] and os.path.exists(dict_path(args['preprocess']['source_lang'])):
    #     raise FileExistsError(dict_path(args['preprocess']['source_lang']))

    if args['preprocess']['srcdict']:
        src_dict = task.load_dictionary(args['preprocess']['srcdict'])
    else:
        assert args['preprocess']['trainpref'], "--trainpref must be set if --srcdict is not specified"
        src_dict = task.build_dictionary(
            [train_path()],
            tokenize_func=get_code_tokens,
            workers=args['preprocess']['workers'],
            threshold=args['preprocess']['thresholdsrc'],
            nwords=args['preprocess']['nwordssrc'],
            # padding_factor=args['preprocess']['padding_factor'],
        )
        src_dict.save_json(dict_path(args['preprocess']['source_lang']))  # save spm dict to ncc.dictionary

    # 2. ***************build dataset********************
    def seperate_tokenize(line):
        tokens = get_code_tokens(line)
        tokens = separate_list(tokens, args['preprocess']['max_len'])
        return tokens

    def make_binary_dataset(vocab: Dictionary, input_file, output_file, num_workers: int):
        """make binary dataset"""
        # LOGGER.info("[{}] Dictionary: {} types".format(attr, len(vocab) - 1))
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
                        vocab,
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
        ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(vocab))

        ext_ds_file = '{}.ext.mmap'.format(output_file)
        ext_ds = indexed_dataset.make_builder(ext_ds_file, impl=args['preprocess']['dataset_impl'],
                                              vocab_size=len(vocab))

        def consumer(data, start_idx):
            ds.add_item(data)
            ext_ds.add_item(start_idx)

        merge_result(
            Binarizer.binarize_seperate(
                input_file, vocab, consumer,
                tokenize=seperate_tokenize, offset=0, end=offsets[1], append_eos=False,
            )
        )
        if num_workers > 1:
            # p1-pN
            pool.join()
            # merge sub-processors' index and data files into final files and delete them.
            for worker_id in range(1, num_workers):
                temp_file_path = "{}{}".format(output_file, worker_id)
                ds.merge_file_(temp_file_path)
                ext_ds.merge_file_(temp_file_path + ".ext")
                # idx, txt
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))
                os.remove(indexed_dataset.data_file_path(temp_file_path + ".ext"))
                os.remove(indexed_dataset.index_file_path(temp_file_path + ".ext"))
        ds.finalize('{}.idx'.format(output_file))
        ext_ds.finalize('{}.ext.idx'.format(output_file))

        LOGGER.info(
            "{}: {} sents, {} tokens, {:.3}% replaced by {}".format(
                # attr,
                input_file,
                n_seq_tok[0],
                n_seq_tok[1],
                100 * sum(replaced.values()) / n_seq_tok[1],
                vocab.unk_word,
            )
        )

    def make_dataset(vocab, input_prefix, output_prefix, lang, num_workers=1):
        if args['preprocess']['dataset_impl'] == "raw":
            in_file = file_name(input_prefix, lang)
            out_dir = args['preprocess']['destdir']
            os.makedirs(out_dir, exist_ok=True)
            LOGGER.info('Copying {} into {}'.format(in_file, out_dir))
            shutil.copy(src=in_file, dst=args['preprocess']['destdir'])
        else:
            in_file = f"{input_prefix}.json"
            out_file = dest_path(output_prefix, lang)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            make_binary_dataset(vocab, in_file, out_file, num_workers)

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

    # for attr in args['preprocess']['source_lang']:
    make_all(args['preprocess']['source_lang'], src_dict)


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--yaml_file", "-f", default='config/preprocess', type=str,
        help="load {yaml_file}.yml for train",
    )
    args = parser.parse_args()
    LOGGER.info(args)
    yaml_file = os.path.join(os.path.dirname(__file__), '{}.yml'.format(args.yaml_file))
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == "__main__":
    cli_main()
