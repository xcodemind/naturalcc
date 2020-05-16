# -*- coding: utf-8 -*-

from typing import *

import os
import re
import ujson
import shutil
import itertools

from ncc import LOGGER
from ncc.utils import tokenizer
from ncc.data import (
    Dictionary,
    constants,
    indexed_dataset,
)
from ncc.data.binarizer import Binarizer
from ncc.multiprocessing import mreader


def train_path(args, lang):
    """get train data path"""
    if args['preprocess']['inserted']:
        return "{}_inserted{}".format(args['preprocess']['trainpref'], ("." + lang) if lang else "")
    else:
        return "{}{}".format(args['preprocess']['trainpref'], ("." + lang) if lang else "")


def insert_sep_token(src_file: str, tgt_file: str):
    """insert function"""
    # insert <S_SEP> to .code files
    with open(src_file, 'r') as reader:
        with open(tgt_file, 'w') as writer:
            for line in reader.readlines():
                ln = ujson.loads(line)
                ln = re.sub('\n\s*\n', '\n', ln)  # remove "\n \n" -> \n
                ln = ln.replace('\n', constants.S_SEP)
                ln = constants.CLS + ln  # profix <CLS>
                writer.write(ujson.dumps(ln) + '\n')


######################################################################
# dictionary functions
######################################################################

def dict_path(args: Dict, modality: str) -> str:
    """Get vocab token file. This file is not dictionary file. Dictionary file will be written later."""

    def _default_path():
        dict_path = os.path.join(os.path.join(args['preprocess']['destdir'], 'dict.{}.txt'.format(modality)))
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
        -> Dictionary:
    """build dictionary for modality"""
    assert src ^ tgt, RuntimeError('Cannot build dictionary for source and target domain at the same time.')
    return task.build_dictionary(
        filenames,
        modality=modality,
        tokenize_func=tokenizer.CSN_tokenizer(modality),
        workers=args['preprocess']['workers'],
        threshold=args['preprocess']['thresholdsrc'] if src else args['preprocess']['thresholdtgt'],
        nwords=args['preprocess']['nwordssrc'] if src else args['preprocess']['nwordstgt'],
        padding_factor=args['preprocess']['padding_factor'],
    )


def load_dict(args: Dict, task, modality: str, overwrite: bool):
    """load dict from (default) dictionary file path. if not exit, load from raw data and save it at default path"""
    dict_filename = dict_path(args, modality)
    if os.path.exists(dict_filename) and (not overwrite):
        LOGGER.info('Dict({}) exists and overwrite=False, skip this.'.format(dict_filename))
        dict = Dictionary.load(dict_filename, attr=modality)
    else:
        # update dict from data
        dict = build_dictionary(args, task, modality, [train_path(args, modality)], src=True)
        # save dict
        LOGGER.info('Save dict(s) for {} in {}.'.format(modality, dict_filename))
        dict.save(dict_filename)
    return dict


######################################################################
# dataset functions
######################################################################


def file_name(file_prefix: str, attr: str, inserted: bool = False) -> str:
    fname = file_prefix + '_inserted' if inserted else file_prefix
    if attr is not None:
        fname += ".{lang}".format(lang=attr)
    return fname


def dest_path(args: Dict, file_prefix: str, attr: str) -> str:
    return os.path.join(args['preprocess']['destdir'], file_name(file_prefix, attr))


def dataset_dest_prefix(args: Dict, file_prefix: str, attr: str):
    """dataset file name without extension"""
    dest_file = os.path.join(args['preprocess']['destdir'], '.'.join([file_prefix, attr]))
    return dest_file


def dataset_dest_file(args: Dict, file_prefix: str, attr: str, extension: str):
    """generate bin file for each modality"""
    base = dataset_dest_prefix(args, file_prefix, attr)
    return "{}.{}".format(base, extension)


def binarize(args: Dict, filename: str, dict: Dictionary, file_prefix: str, attr: str,
             offset: int, end: int, append_eos: bool = True):
    """binarize function for multi-processing"""
    ds_file = dataset_dest_file(args, file_prefix, attr, extension='bin')
    ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(dict))

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize(filename, dict, consumer, tokenize=tokenizer.CSN_tokenizer(attr),
                             append_eos=append_eos, offset=offset, end=end)
    ds.finalize(dataset_dest_file(args, file_prefix, attr, extension="idx"))
    return res


def make_binary_dataset(args: Dict, dict: Dictionary, input_prefix, output_prefix,
                        attr: str, num_workers: int, inserted: bool = False):
    """make binary dataset"""
    LOGGER.info("[{}] Dictionary: {} types".format(attr, len(dict) - 1))
    n_seq_tok = [0, 0]
    replaced = Counter()  # save un-recorded tokens

    def merge_result(worker_result):
        replaced.update(worker_result["replaced"])
        n_seq_tok[0] += worker_result["nseq"]
        n_seq_tok[1] += worker_result["ntok"]

    input_prefix = input_prefix + '_inserted' if inserted else input_prefix
    input_file = '{}.{}'.format(input_prefix, attr)
    # split a file into different parts
    # if use multi-processing, we first process 2nd to last file
    # 1.txt -> 10 processor, 0(p0)(0-99), 100(p1)(100-199), ...
    offsets = Binarizer.find_offsets(input_file, num_workers)
    pool = None
    if num_workers > 1:
        # p1-pN -> (1 bin-txt, 1 idx), (N bin-txt, N idx)
        pool = Pool(processes=num_workers - 1)
        for worker_id in range(1, num_workers):
            prefix = "{}{}".format(output_prefix, worker_id)
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
    ds_file = dataset_dest_file(args, output_prefix, attr, extension='bin')
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
            prefix = "{}{}".format(output_prefix, worker_id)
            temp_file_path = dataset_dest_prefix(args, prefix, attr)
            ds.merge_file_(temp_file_path)
            # idx, txt
            os.remove(indexed_dataset.data_file_path(temp_file_path))
            os.remove(indexed_dataset.index_file_path(temp_file_path))

    ds.finalize(dataset_dest_file(args, output_prefix, attr, extension="idx"))

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


def make_dataset(args: Dict, dict: Dictionary, input_prefix, output_prefix, attr: str,
                 num_workers: int = 1):
    """
    build raw/bin dataset
    1) raw dataset: copy raw files
    2) bin dataset: build bin files
    """
    if args['preprocess']['dataset_impl'] == "raw":
        # Copy original text file to destination folder
        output_text_file = dest_path(args, output_prefix, attr, )
        shutil.copyfile(file_name(input_prefix, attr), output_text_file)
    else:
        make_binary_dataset(args, dict, input_prefix, output_prefix, attr, num_workers)


def make_all(args: Dict, attr: str, dict: Dictionary, modes: Optional[List] = None):
    """build dataset with dictionary for [train/valid/test] mode"""
    if modes is None:
        modes = constants.MODES
    for mode in modes:
        file_prefix = args['preprocess']['{}pref'.format(mode)]
        if file_prefix:
            make_dataset(args, dict, file_prefix, mode, attr, num_workers=args['preprocess']['workers'])


def path_special_symbols(files: List[str]) -> Set:
    """get special symbols of path for bert bpe encoding"""
    special_symbols = set()

    def path_body_tokens(line: str, *args, **kwargs):
        line = ujson.loads(line.strip())
        paths = line[len(constants.CLS):].split(constants.S_SEP)
        body = [tokenizer.tokenize_string(
            re.split(r'{}|{}'.format(constants.H_SEP, constants.T_SEP), path)[1]
        ) for path in paths]
        return set(itertools.chain(*body))

    for file in files:
        print(file)
        mtokens = mreader.readline(file, path_body_tokens)
        for tokens in itertools.chain(*mtokens):
            special_symbols.update(tokens)
    return special_symbols
