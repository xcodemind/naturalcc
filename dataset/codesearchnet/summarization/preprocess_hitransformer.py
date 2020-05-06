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
from ncc.utils.util_file import load_yaml
from ncc.utils import utils
from ncc import LOGGER
from .preprocess_helper import *
# logging.basicConfig(
#     format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S',
#     level=logging.INFO,
#     stream=sys.stdout,
# )
# logger = logging.getLogger('preprocess')


def parse_alignment(line):
    """
    Parses a single line from the alingment file.

    Args:
        line (str): String containing the alignment of the format:
            <src_idx_1>-<tgt_idx_1> <src_idx_2>-<tgt_idx_2> ..
            <src_idx_m>-<tgt_idx_m>. All indices are 0 indexed.

    Returns:
        torch.IntTensor: packed alignments of shape (2 * m).
    """
    alignments = line.strip().split()
    parsed_alignment = torch.IntTensor(2 * len(alignments))
    for idx, alignment in enumerate(alignments):
        src_idx, tgt_idx = alignment.split("-")
        parsed_alignment[2 * idx] = int(src_idx)
        parsed_alignment[2 * idx + 1] = int(tgt_idx)
    return parsed_alignment


def train_path(args, lang):
    return "{}{}".format(args['preprocess']['trainpref'], ("." + lang) if lang else "")

def file_name(prefix, lang):
    fname = prefix
    if lang is not None:
        fname += ".{lang}".format(lang=lang)
    return fname


def dest_path(args, prefix, lang):
    return os.path.join(args['preprocess']['destdir'], file_name(prefix, lang))


def dict_path(args, lang):
    return dest_path(args, "dict", lang) + ".txt"


def build_dictionary(args, task, modality, filenames, src=False, tgt=False):
    assert src ^ tgt
    return task.build_dictionary(
        filenames,
        modality=modality,
        workers=args['preprocess']['workers'],
        threshold=args['preprocess']['thresholdsrc'] if src else args['preprocess']['thresholdtgt'],
        nwords=args['preprocess']['nwordssrc'] if src else args['preprocess']['nwordstgt'],
        padding_factor=args['preprocess']['padding_factor'],
    )


def make_binary_dataset(args, vocab, input_prefix, output_prefix, lang, num_workers):
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


def make_binary_alignment_dataset(args, input_prefix, output_prefix, num_workers):
    nseq = [0]

    def merge_result(worker_result):
        nseq[0] += worker_result['nseq']

    input_file = input_prefix
    offsets = Binarizer.find_offsets(input_file, num_workers)
    pool = None
    if num_workers > 1:
        pool = Pool(processes=num_workers - 1)
        for worker_id in range(1, num_workers):
            prefix = "{}{}".format(output_prefix, worker_id)
            pool.apply_async(
                binarize_alignments,
                (
                    args,
                    input_file,
                    utils.parse_alignment,
                    prefix,
                    offsets[worker_id],
                    offsets[worker_id + 1]
                ),
                callback=merge_result
            )
        pool.close()

    ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, None, "bin"),
                                      impl=args['preprocess']['dataset_impl'])

    merge_result(
        Binarizer.binarize_alignments(
            input_file, utils.parse_alignment, lambda t: ds.add_item(t),
            offset=0, end=offsets[1]
        )
    )
    if num_workers > 1:
        pool.join()
        for worker_id in range(1, num_workers):
            prefix = "{}{}".format(output_prefix, worker_id)
            temp_file_path = dataset_dest_prefix(args, prefix, None)
            ds.merge_file_(temp_file_path)
            os.remove(indexed_dataset.data_file_path(temp_file_path))
            os.remove(indexed_dataset.index_file_path(temp_file_path))

    ds.finalize(dataset_dest_file(args, output_prefix, None, "idx"))

    LOGGER.info(
        "[alignments] {}: parsed {} alignments".format(
            input_file,
            nseq[0]
        )
    )



def make_dataset(args, vocab, input_prefix, output_prefix, lang, num_workers=1):
    if args['preprocess']['dataset_impl'] == "raw":
        # Copy original text file to destination folder
        output_text_file = dest_path(args,
            output_prefix, # + ".{}-{}".format(args['preprocess']['source_lang'], args['preprocess']['target_lang'])
            lang,
        )
        shutil.copyfile(file_name(input_prefix, lang), output_text_file)
    else:
        make_binary_dataset(args, vocab, input_prefix, output_prefix, lang, num_workers)


def make_all(args, lang, vocab):
    if args['preprocess']['trainpref']:
        make_dataset(args, vocab, args['preprocess']['trainpref'], "train", lang, num_workers=args['preprocess']['workers'])
    if args['preprocess']['validpref']:
        for k, validpref in enumerate(args['preprocess']['validpref'].split(",")):
            outprefix = "valid{}".format(k) if k > 0 else "valid"
            make_dataset(args, vocab, validpref, outprefix, lang, num_workers=args['preprocess']['workers'])
    if args['preprocess']['testpref']:
        for k, testpref in enumerate(args['preprocess']['testpref'].split(",")):
            outprefix = "test{}".format(k) if k > 0 else "test"
            make_dataset(args, vocab, testpref, outprefix, lang, num_workers=args['preprocess']['workers'])


def make_all_alignments(args):
    if args['preprocess']['trainpref'] and os.path.exists(args['preprocess']['trainpref'] + "." + args['preprocess']['align_suffix']):
        make_binary_alignment_dataset(args, args['preprocess']['trainpref'] + "." + args['preprocess']['align_suffix'], "train.align", num_workers=args['preprocess']['workers'])
    if args['preprocess']['validpref'] and os.path.exists(args['preprocess']['validpref'] + "." + args['preprocess']['align_suffix']):
        make_binary_alignment_dataset(args, args['preprocess']['validpref'] + "." + args['preprocess']['align_suffix'], "valid.align", num_workers=args['preprocess']['workers'])
    if args['preprocess']['testpref'] and os.path.exists(args['preprocess']['testpref'] + "." + args['preprocess']['align_suffix']):
        make_binary_alignment_dataset(args, args['preprocess']['testpref'] + "." + args['preprocess']['align_suffix'], "test.align", num_workers=args['preprocess']['workers'])


def binarize(args, filename, vocab, output_prefix, lang, offset, end, append_eos=True):
    ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, lang, "bin"),
                                      impl=args['preprocess']['dataset_impl'], vocab_size=len(vocab))

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize(filename, vocab, consumer, append_eos=append_eos,
                             offset=offset, end=end)
    ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
    return res


def binarize_alignments(args, filename, parse_alignment, output_prefix, offset, end):
    ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, None, "bin"),
                                      impl=args['preprocess']['dataset_impl'], vocab_size=None)

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize_alignments(filename, parse_alignment, consumer, offset=offset,
                                        end=end)
    ds.finalize(dataset_dest_file(args, output_prefix, None, "idx"))
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


def get_offsets(input_file, num_workers):
    return Binarizer.find_offsets(input_file, num_workers)


def main(args):
    # utils.import_user_module(args)
    os.makedirs(args['preprocess']['destdir'], exist_ok=True)
    target = not args['preprocess']['only_source']

    # if not args['preprocess']['codedict'] and os.path.exists(dict_path(args, args['preprocess']['source_lang'])):
    #     raise FileExistsError(dict_path(args, args['preprocess']['source_lang']))
    # if target and not args['preprocess']['tgtdict'] and os.path.exists(dict_path(args, args['preprocess']['target_lang'])):
    #     raise FileExistsError(dict_path(args, args['preprocess']['target_lang']))
    modality = args['preprocess']['source_lang']

    # 0. insert special token
    insert_sep_tokens(args)

    # 1. Build vocabulary (dictionary)
    LOGGER.info('Build vocabulary...')
    task = tasks.get_task(args['preprocess']['task'])
    if args['preprocess']['joined_dictionary']:     # TODO: to be checked
        assert not args['preprocess']['codedict'] or not args['preprocess']['tgtdict'], \
            "cannot use both --srcdict and --tgtdict with --joined-dictionary"

        assert args['preprocess']['trainpref'], "--trainpref must be set if --codedict is not specified"
        src_dict = build_dictionary(args, task,
            {train_path(args, lang) for lang in args['preprocess']['source_lang'] + args['preprocess']['target_lang']}, src=True
        )
        tgt_dict = src_dict
    else:
        # if args['preprocess']['codedict']:
        #     src_dict = task.load_dictionary(args['preprocess']['codedict'])
        # else:
        # src_dicts = OrderedDict()
        # for modality in args['preprocess']['source_lang']:
        assert args['preprocess']['trainpref'], "--trainpref must be set if --codedict is not specified"
        if modality == 'path':
            src_dict1, src_dict2 = build_dictionary(args, task, modality, [train_path(args, modality)], src=True)
            # src_dict = build_dictionary(train_path(args['preprocess']['source_lang']), src=True)
            src_dict = [src_dict1, src_dict2]
        else:
            src_dict = build_dictionary(args, task, modality, [train_path(args, modality)], src=True)
            # src_dict = build_dictionary(train_path(args['preprocess']['source_lang']), src=True)
            # src_dicts[modality] = src_dict

        if target:
            if args['preprocess']['tgtdict']:
                tgt_dict = task.load_dictionary(args['preprocess']['tgtdict'])
            else:
                assert args['preprocess']['trainpref'], "--trainpref must be set if --tgtdict is not specified"
                tgt_dict = build_dictionary(args, task, modality, [train_path(args, args['preprocess']['target_lang'])], tgt=True)
                # tgt_dict = build_dictionary(train_path(args['preprocess']['target_lang']), tgt=True)

        else:
            tgt_dict = None
    LOGGER.info('Save vocabulary.')
    # for modality in args['preprocess']['source_lang']:
    if modality == 'path':
        src_dict[0].save(dict_path(args, modality+'_border'))
        src_dict[1].save(dict_path(args, modality + '_center'))
    else:
        src_dict.save(dict_path(args, modality))
    if target and tgt_dict is not None:
        tgt_dict.save(dict_path(args, args['preprocess']['target_lang']))

    # sys.exit()
    # 2. Make dataset (raw or mmap..)
    LOGGER.info('Make dataset...')
    # make_all(args, args['preprocess']['source_lang'], src_dict) # TODO: source_lang -> modalities
    make_all(args, 'code', src_dict)
    # make_all(args, 'path', src_dict)
    # make_all(args, 'bin_ast', src_dict)
    # make_all(args, 'sbt', src_dict)
    if target:
        make_all(args, args['preprocess']['target_lang'], tgt_dict)
    if args['preprocess']['align_suffix']:
        make_all_alignments(args)

    LOGGER.info("Wrote preprocessed data to {}".format(args['preprocess']['destdir']))

    if args['preprocess']['alignfile']:
        assert args['preprocess']['trainpref'], "--trainpref must be set if --alignfile is specified"
        src_file_name = train_path(args, args['preprocess']['source_lang'])
        tgt_file_name = train_path(args, args['preprocess']['target_lang'])
        freq_map = {}
        with open(args['preprocess']['alignfile'], "r", encoding='utf-8') as align_file:
            with open(src_file_name, "r", encoding='utf-8') as src_file:
                with open(tgt_file_name, "r", encoding='utf-8') as tgt_file:
                    for a, s, t in zip_longest(align_file, src_file, tgt_file):
                        si = src_dict.encode_line(s, add_if_not_exist=False)
                        ti = tgt_dict.encode_line(t, add_if_not_exist=False)
                        ai = list(map(lambda x: tuple(x.split("-")), a.split()))
                        for sai, tai in ai:
                            srcidx = si[int(sai)]
                            tgtidx = ti[int(tai)]
                            if srcidx != src_dict.unk() and tgtidx != tgt_dict.unk():

                                if srcidx not in freq_map:
                                    freq_map[srcidx] = {}
                                if tgtidx not in freq_map[srcidx]:
                                    freq_map[srcidx][tgtidx] = 1
                                else:
                                    freq_map[srcidx][tgtidx] += 1

        align_dict = {}
        for srcidx in freq_map.keys():
            align_dict[srcidx] = max(freq_map[srcidx], key=freq_map[srcidx].get)

        with open(
                os.path.join(
                    args['preprocess']['destdir'],
                    "alignment.{}-{}.txt".format(args['preprocess']['source_lang'], args['preprocess']['target_lang']),
                ),
                "w", encoding='utf-8'
        ) as f:
            for k, v in align_dict.items():
                print("{} {}".format(src_dict[k], tgt_dict[v]), file=f)


def cli_main():
    Argues = namedtuple('Argues', 'yaml')

    args_ = Argues('preprocess_hitransformer.yml')  # train_sl
    LOGGER.info(args_)
    # print(type(args.multi_processing))
    # assert False
    print('args: ', type(args_))
    # config = run_init(args.yaml, config=None)
    yaml_file = os.path.join(sys.path[0], args_.yaml)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)

    LOGGER.info(args)
    main(args)


if __name__ == "__main__":
    cli_main()

