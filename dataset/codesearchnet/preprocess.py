#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

from collections import Counter
from itertools import zip_longest
import logging
from multiprocessing import Pool
import os
import shutil
import sys
# sys.path.append('../..')
from ncc import tasks #, utils #  options,
# from fairseq.data import indexed_dataset
from ncc.data import indexed_dataset
from ncc.data.binarizer import Binarizer
import argparse
from ncc.data.indexed_dataset import get_available_dataset_impl
import torch
from ncc.utils import utils
import os
import sys
import math
import random
import numpy as np
from collections import namedtuple
import torch
from ncc import LOGGER
from ncc import tasks
from ncc.logging import meters
from ncc.trainer.fair_trainer import Trainer
from ncc.utils import checkpoint_utils, distributed_utils
from ncc.utils.util_file import load_yaml
from ncc.logging import metrics, progress_bar
from ncc.utils import utils
from ncc.data import iterators
import glob

# logging.basicConfig(
#     format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S',
#     level=logging.INFO,
#     stream=sys.stdout,
# )
# logger = logging.getLogger('preprocess')
from ncc import LOGGER

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

def main(args):
    # utils.import_user_module(args)
    os.makedirs(args['preprocess']['destdir'], exist_ok=True)

    # logger.addHandler(logging.FileHandler(
    #     filename=os.path.join(args.destdir, 'preprocess.log'),
    # ))
    # logger.info(args)
    print('args.task: ', args['preprocess']['task'])
    task = tasks.get_task(args['preprocess']['task'])

    def train_path(lang):
        # return "{}{}".format(args['preprocess']['trainpref'], ("." + lang) if lang else "")
        paths = []
        if lang == 'code':
            for file in glob.glob(os.path.join(args['preprocess']['trainpref'], 'code', '*train*.txt')):
                paths.append(os.path.join(args['preprocess']['trainpref'], file))
        elif lang == 'comment':
            for file in glob.glob(os.path.join(args['preprocess']['trainpref'], 'docstring', '*train*.txt')):
                paths.append(os.path.join(args['preprocess']['trainpref'], file))
        return paths

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
            threshold=args['preprocess']['thresholdsrc'] if src else args['preprocess']['thresholdtgt'],
            nwords=args['preprocess']['nwordssrc'] if src else args['preprocess']['nwordstgt'],
            padding_factor=args['preprocess']['padding_factor'],
        )

    target = not args['preprocess']['only_source']

    if not args['preprocess']['srcdict'] and os.path.exists(dict_path(args['preprocess']['source_lang'])):
        raise FileExistsError(dict_path(args['preprocess']['source_lang']))
    if target and not args['preprocess']['tgtdict'] and os.path.exists(dict_path(args['preprocess']['target_lang'])):
        raise FileExistsError(dict_path(args['preprocess']['target_lang']))

    if args['preprocess']['joined_dictionary']:
        assert not args['preprocess']['srcdict'] or not args['preprocess']['tgtdict'], \
            "cannot use both --srcdict and --tgtdict with --joined-dictionary"

        if args['preprocess']['srcdict']:
            src_dict = task.load_dictionary(args['preprocess']['srcdict'])
        elif args['preprocess']['tgtdict']:
            src_dict = task.load_dictionary(args['preprocess']['tgtdict'])
        else:
            assert args['preprocess']['trainpref'], "--trainpref must be set if --srcdict is not specified"
            src_dict = build_dictionary(
                {train_path(lang) for lang in [args['preprocess']['source_lang'], args['preprocess']['target_lang']]}, src=True
            )
        tgt_dict = src_dict
    else:
        if args['preprocess']['srcdict']:
            src_dict = task.load_dictionary(args['preprocess']['srcdict'])
        else:
            assert args['preprocess']['trainpref'], "--trainpref must be set if --srcdict is not specified"
            # src_dict = build_dictionary([train_path(args['preprocess']['source_lang'])], src=True)
            src_dict = build_dictionary(train_path(args['preprocess']['source_lang']), src=True)

        if target:
            if args['preprocess']['tgtdict']:
                tgt_dict = task.load_dictionary(args['preprocess']['tgtdict'])
            else:
                assert args['preprocess']['trainpref'], "--trainpref must be set if --tgtdict is not specified"
                # tgt_dict = build_dictionary([train_path(args['preprocess']['target_lang'])], tgt=True)
                tgt_dict = build_dictionary(train_path(args['preprocess']['target_lang']), tgt=True)

        else:
            tgt_dict = None

    src_dict.save(dict_path(args['preprocess']['source_lang']))
    if target and tgt_dict is not None:
        tgt_dict.save(dict_path(args['preprocess']['target_lang']))

    # sys.exit()

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

    def make_binary_alignment_dataset(input_prefix, output_prefix, num_workers):
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

    def make_dataset(vocab, input_prefix, output_prefix, lang, num_workers=1):
        if args['preprocess']['dataset_impl'] == "raw":
            # Copy original text file to destination folder
            output_text_file = dest_path(
                output_prefix + ".{}-{}".format(args['preprocess']['source_lang'], args['preprocess']['target_lang']),
                lang,
            )
            shutil.copyfile(file_name(input_prefix, lang), output_text_file)
        else:
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

    def make_all_alignments():
        if args['preprocess']['trainpref'] and os.path.exists(args['preprocess']['trainpref'] + "." + args['preprocess']['align_suffix']):
            make_binary_alignment_dataset(args['preprocess']['trainpref'] + "." + args['preprocess']['align_suffix'], "train.align", num_workers=args['preprocess']['workers'])
        if args['preprocess']['validpref'] and os.path.exists(args['preprocess']['validpref'] + "." + args['preprocess']['align_suffix']):
            make_binary_alignment_dataset(args['preprocess']['validpref'] + "." + args['preprocess']['align_suffix'], "valid.align", num_workers=args['preprocess']['workers'])
        if args['preprocess']['testpref'] and os.path.exists(args['preprocess']['testpref'] + "." + args['preprocess']['align_suffix']):
            make_binary_alignment_dataset(args['preprocess']['testpref'] + "." + args['preprocess']['align_suffix'], "test.align", num_workers=args['preprocess']['workers'])

    make_all(args['preprocess']['source_lang'], src_dict)
    if target:
        make_all(args['preprocess']['target_lang'], tgt_dict)
    if args['preprocess']['align_suffix']:
        make_all_alignments()

    LOGGER.info("Wrote preprocessed data to {}".format(args['preprocess']['destdir']))

    if args['preprocess']['alignfile']:
        assert args['preprocess']['trainpref'], "--trainpref must be set if --alignfile is specified"
        src_file_name = train_path(args['preprocess']['source_lang'])
        tgt_file_name = train_path(args['preprocess']['target_lang'])
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
                                assert srcidx != src_dict.pad()
                                assert srcidx != src_dict.eos()
                                assert tgtidx != tgt_dict.pad()
                                assert tgtidx != tgt_dict.eos()

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


# def cli_main():
#     parser = options.get_preprocessing_parser()
#     args = parser.parse_args()
#     main(args)
def cli_main():
    Argues = namedtuple('Argues', 'yaml')

    args_ = Argues('preprocess.yml')  # train_sl
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
    # sys.exit()
    # # cli_main()
    # # parser = get_parser("Preprocessing", default_task)
    # parser = argparse.ArgumentParser()
    # # Before creating the true parser, we need to import optional user module
    # # in order to eagerly import custom tasks, optimizers, architectures, etc.
    # usr_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    # usr_parser.add_argument("--user-dir", default=None)
    # usr_args, _ = usr_parser.parse_known_args()
    # # utils.import_user_module(usr_args)
    #
    # parser = argparse.ArgumentParser(allow_abbrev=False)
    # # fmt: off
    # parser.add_argument('--no-progress-bar', action='store_true', help='disable progress bar')
    # parser.add_argument('--log-interval', type=int, default=100, metavar='N',
    #                     help='log progress every N batches (when progress bar is disabled)')
    # parser.add_argument('--log-format', default=None, help='log format to use',
    #                     choices=['json', 'none', 'simple', 'tqdm'])
    # parser.add_argument('--tensorboard-logdir', metavar='DIR', default='',
    #                     help='path to save logs for tensorboard, should match --logdir '
    #                          'of running tensorboard (default: no tensorboard logging)')
    # parser.add_argument('--seed', default=1, type=int, metavar='N',
    #                     help='pseudo random number generator seed')
    # parser.add_argument('--cpu', action='store_true', help='use CPU instead of CUDA')
    # parser.add_argument('--fp16', action='store_true', help='use FP16')
    # parser.add_argument('--memory-efficient-fp16', action='store_true',
    #                     help='use a memory-efficient version of FP16 training; implies --fp16')
    # parser.add_argument('--fp16-no-flatten-grads', action='store_true',
    #                     help='don\'t flatten FP16 grads tensor')
    # parser.add_argument('--fp16-init-scale', default=2 ** 7, type=int,
    #                     help='default FP16 loss scale')
    # parser.add_argument('--fp16-scale-window', type=int,
    #                     help='number of updates before increasing loss scale')
    # parser.add_argument('--fp16-scale-tolerance', default=0.0, type=float,
    #                     help='pct of updates that can overflow before decreasing the loss scale')
    # parser.add_argument('--min-loss-scale', default=1e-4, type=float, metavar='D',
    #                     help='minimum FP16 loss scale, after which training is stopped')
    # parser.add_argument('--threshold-loss-scale', type=float,
    #                     help='threshold FP16 loss scale from below')
    # parser.add_argument('--user-dir', default=None,
    #                     help='path to a python module containing custom extensions (tasks and/or architectures)')
    # parser.add_argument('--empty-cache-freq', default=0, type=int,
    #                     help='how often to clear the PyTorch CUDA cache (0 to disable)')
    # parser.add_argument('--all-gather-list-size', default=16384, type=int,
    #                     help='number of bytes reserved for gathering stats from workers')
    #
    # from ncc.registry import REGISTRIES
    #
    # for registry_name, REGISTRY in REGISTRIES.items():
    #     parser.add_argument(
    #         '--' + registry_name.replace('_', '-'),
    #         default=REGISTRY['default'],
    #         choices=REGISTRY['registry'].keys(),
    #     )
    #
    # # Task definitions can be found under fairseq/tasks/
    # from ncc.tasks import TASK_REGISTRY
    #
    # parser.add_argument('--task', metavar='TASK', default="translation",
    #                     choices=TASK_REGISTRY.keys(),
    #                     help='task')
    # # fmt: on
    #
    # # ===========================
    # group = parser.add_argument_group("Preprocessing")
    # # fmt: off
    # group.add_argument("-s", "--source-lang", default=None, metavar="SRC",
    #                    help="source language")
    # group.add_argument("-t", "--target-lang", default=None, metavar="TARGET",
    #                    help="target language")
    # group.add_argument("--trainpref", metavar="FP", default=None,
    #                    help="train file prefix")
    # group.add_argument("--validpref", metavar="FP", default=None,
    #                    help="comma separated, valid file prefixes")
    # group.add_argument("--testpref", metavar="FP", default=None,
    #                    help="comma separated, test file prefixes")
    # group.add_argument("--align-suffix", metavar="FP", default=None,
    #                    help="alignment file suffix")
    # group.add_argument("--destdir", metavar="DIR", default="data-bin",
    #                    help="destination dir")
    # group.add_argument("--thresholdtgt", metavar="N", default=0, type=int,
    #                    help="map words appearing less than threshold times to unknown")
    # group.add_argument("--thresholdsrc", metavar="N", default=0, type=int,
    #                    help="map words appearing less than threshold times to unknown")
    # group.add_argument("--tgtdict", metavar="FP",
    #                    help="reuse given target dictionary")
    # group.add_argument("--srcdict", metavar="FP",
    #                    help="reuse given source dictionary")
    # group.add_argument("--nwordstgt", metavar="N", default=-1, type=int,
    #                    help="number of target words to retain")
    # group.add_argument("--nwordssrc", metavar="N", default=-1, type=int,
    #                    help="number of source words to retain")
    # group.add_argument("--alignfile", metavar="ALIGN", default=None,
    #                    help="an alignment file (optional)")
    # parser.add_argument('--dataset-impl', metavar='FORMAT', default='mmap',
    #                     choices=get_available_dataset_impl(),
    #                     help='output dataset implementation')
    # group.add_argument("--joined-dictionary", action="store_true",
    #                    help="Generate joined dictionary")
    # group.add_argument("--only-source", action="store_true",
    #                    help="Only process the source language")
    # group.add_argument("--padding-factor", metavar="N", default=8, type=int,
    #                    help="Pad dictionary size to be multiple of N")
    # group.add_argument("--workers", metavar="N", default=1, type=int,
    #                    help="number of parallel workers")
    # args = parser.parse_args()
    # main(args)
