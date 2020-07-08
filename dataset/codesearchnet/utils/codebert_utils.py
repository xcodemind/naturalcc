# -*- coding: utf-8 -*-

from typing import *

import os
import re
import sys
import ujson
import shutil
import itertools
import contextlib
import sentencepiece as spm
from multiprocessing import Pool

from ncc import LOGGER
from ncc.utils import tokenizer
from ncc.data import (
    Dictionary,
    constants,
    indexed_dataset,
)
from ncc.data.tools.binarizer import Binarizer
from ncc.data.constants import (PAD, EOS, UNK, BOS)
import ncc.utils.mp_mreader as mreader

DICTIONARY_CONTAINED = set([PAD, EOS, UNK, BOS])


def train_path(args, lang):
    """get train data path"""
    if args['preprocess']['inserted']:
        return "{}_inserted{}".format(args['preprocess']['trainpref'], ("." + lang) if lang else "")
    else:
        return "{}{}".format(args['preprocess']['trainpref'], ("." + lang) if lang else "")


def insert_sep_token(input_file: str, output_file: Optional[str] = None, overwrite: bool = True):
    """insert CLS/S_SEP for bert"""
    if output_file is None:
        input_file = input_file + '_inserted'
    if os.path.exists(output_file) and not overwrite:
        return
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as writer:
        with open(input_file, 'r') as reader:
            for line in reader.readlines():
                ln = ujson.loads(line)
                ln = re.sub('\n\s*\n', '\n', ln)  # remove "\n \n" -> \n
                ln = ln.replace('\n', ' ' + constants.S_SEP + ' ')  # should be whitespace before and after S_SEP
                # ln = constants.CLS + ' ' + ln  # profix <CLS>, should be whitespace after the CLS
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
        dict = Dictionary.load(dict_filename)
    else:
        # update dict from data
        dict = build_dictionary(args, task, modality, [train_path(args, modality)], src=True)
        # save dict
        LOGGER.info('Save dict(s) for {} in {}.'.format(modality, dict_filename))
        dict.save(dict_filename)
    return dict


def vocab2dict(vocab_file: str, dict_file: str = None):
    if dict_file is None:
        dict_file = vocab_file.replace('.vocab', '.dict')
    with open(vocab_file, 'r', encoding='UTF-8') as reader:
        with open(dict_file, 'w', encoding='UTF-8') as writer:
            for line in reader:
                token, _ = line.rstrip('\n').split('\t')
                if token in DICTIONARY_CONTAINED:
                    continue
                writer.write('{} {}\n'.format(token, 1))


######################################################################
# dataset functions
######################################################################


def binarize(args: Dict, filename: str, dict: Dictionary, in_file: str, attr: str,
             offset: int, end: int, append_eos: bool = True):
    """binarize function for multi-processing"""
    ds_file = '{}.bin'.format(in_file)
    ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(dict))

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize(filename, dict, consumer, tokenize=tokenizer.CSN_tokenizer(attr),
                             append_eos=append_eos, offset=offset, end=end)
    ds.finalize('{}.idx'.format(in_file))
    return res


def make_binary_dataset(args: Dict, dict: Dictionary, input_file, output_file,
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


def make_all(args: Dict, attr: str, dict: Dictionary, modes: Optional[List] = None):
    """
    build dataset with dictionary for [train/valid/test] mode

    build raw/bin dataset
    1) raw dataset: copy raw files
    2) bin dataset: build bin files
    """
    if modes is None:
        modes = constants.MODES
    for mode in modes:
        src_file = os.path.join(args['preprocess']['{}pref'.format(mode)], '.'.join([mode, attr]))
        tgt_file = os.path.join(args['preprocess']['destdir'], '.'.join([mode, attr]))
        if args['preprocess']['dataset_impl'] == "raw":
            # Copy original text file to destination folder
            shutil.copyfile(src_file, tgt_file)
        else:
            make_binary_dataset(args, dict, src_file, tgt_file, attr, args['preprocess']['workers'])


def path_special_symbols(files: List[str]) -> Set:
    """get special symbols of path for bert bpe encoding"""
    special_symbols = set()

    def path_body_tokens(line: str, *args, **kwargs):
        line = ujson.loads(line.strip())
        paths = line.split(constants.S_SEP)
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


def get_special_symbols(args: Dict) -> Optional[Set]:
    """some modality need special symbols"""

    def _special_symbols(args: Dict, modality: str) -> Optional[Set]:
        special_symbols = set()
        if modality in ['code']:
            special_symbols.update([constants.S_SEP, constants.CLS])
        elif modality in ['path']:
            special_symbols.update([constants.H_SEP, constants.T_SEP, constants.S_SEP])
            special_symbols.update(path_special_symbols(args.input_files[modality]))
        else:
            return None
        return special_symbols

    def _get_special_symbols(args: Dict, modality: str) -> Optional[Set]:
        default_special_symbols_files = os.path.join(args.tgt_dir, '.{}.ss'.format(modality))
        if os.path.exists(default_special_symbols_files) and (not args.overwrite):
            with open(default_special_symbols_files, 'r') as reader:
                special_symbols = [line.rstrip('\n') for line in reader.readlines()]
        else:
            special_symbols = _special_symbols(args, modality)
            if special_symbols is None:
                return special_symbols
            os.makedirs(os.path.dirname(default_special_symbols_files), exist_ok=True)
            with open(default_special_symbols_files, 'w') as writer:
                for symbol in special_symbols:
                    writer.write(symbol + '\n')
        return special_symbols

    special_symbols = set()
    for modality in args.modalities:
        sp_symbols = _get_special_symbols(args, modality)
        if sp_symbols is None:
            pass
        else:
            special_symbols.update(sp_symbols)
    return special_symbols


# def combine_special_symbols(tokens: List, special_symbols: Optional[Set]) -> List:
#     """merge _ and special_symbols, e.g. _ <CLS> => _<CLS>"""
#     if special_symbols is None:
#         return tokens
#     new_tokens = []
#     idx = 0
#     while idx < len(tokens) - 1:
#         if (tokens[idx] == constants.SP_SPACE) and (tokens[idx + 1] in special_symbols):
#             new_tokens.append(tokens[idx] + tokens[idx + 1])
#             idx += 2
#         else:
#             new_tokens.append(tokens[idx])
#             idx += 1
#     if idx == len(tokens):
#         new_tokens.append(tokens[-1])
#     return new_tokens

def combine_special_symbols(tokens: List, special_symbols: Optional[Set]) -> List:
    """merge _ and special_symbols, e.g. _ <CLS> => _<CLS>"""
    if special_symbols is None:
        return tokens
    new_tokens = []
    idx = 0
    while idx < len(tokens) - 1:
        if (tokens[idx] == constants.SP_SPACE) and (not tokens[idx + 1][0] == constants.SP_SPACE):
            new_tokens.append(tokens[idx] + tokens[idx + 1])
            idx += 2
        else:
            new_tokens.append(tokens[idx])
            idx += 1
    if idx == len(tokens):
        new_tokens.append(tokens[-1])
    return new_tokens


def build_model(file: str, model_name: str, vocab_size: int, special_symbols: Optional[Set] = None,
                overwrite: bool = False):
    os.makedirs(os.path.dirname(model_name), exist_ok=True)
    if os.path.exists('{}.model'.format(model_name)) and os.path.exists('{}.vocab'.format(model_name)) \
            and not overwrite:
        return
    params = '--input={} --model_prefix={} --hard_vocab_limit=false'.format(','.join(file), model_name)
    if special_symbols is not None:
        # PAD, EOS, UNK, BOS is automatically added into Dictionary building, so narrow vocab_size
        vocab_size -= len(special_symbols & DICTIONARY_CONTAINED)
        params += ' --user_defined_symbols={}'.format(','.join(special_symbols))
    params += ' --vocab_size={}'.format(vocab_size)

    LOGGER.info(params)
    spm.SentencePieceTrainer.train(params)
    vocab2dict(vocab_file='{}.vocab'.format(model_name))


class WordpieceEncoder(object):

    def __init__(self, args):
        self.args = args

    def initializer(self):
        global sp
        # bpe = get_encoder(self.args.encoder_json, self.args.vocab_bpe)
        sp = spm.SentencePieceProcessor()
        sp.Load('{}.model'.format(self.args.bpe_models))

    def encode(self, line):
        global sp
        if self.args.format == 'id':
            ids = sp.EncodeAsIds(line)
            return list(map(str, ids))
        elif self.args.format == 'piece':
            pieces = sp.EncodeAsPieces(line)
            return pieces

    def decode(self, tokens):
        global sp
        # return bpe.decode(tokens)
        if self.args.format == 'id':
            return sp.DecodeIds(tokens)
        elif self.args.format == 'piece':
            return sp.DecodePieces(tokens)

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = line.strip()
            line = ujson.loads(line)
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None]
            # line = ' "_operator '
            tokens = self.encode(line)
            # tokens = combine_special_symbols(tokens, self.args.special_symbols)
            enc_lines.append(" ".join(tokens))
        return ["PASS", enc_lines]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]


def write_bpe_files(args: Dict, input_file: List[str], output_file: List[str]):
    assert len(input_file) == len(output_file) == 1, \
        RuntimeError("number of input and output paths should match with 1")

    with contextlib.ExitStack() as stack:
        input_files = [
            stack.enter_context(open(input, "r", encoding="utf-8"))
            if input != "-" else sys.stdin
            for input in input_file
        ]
        output_files = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-" else sys.stdout
            for output in output_file
        ]

        encoder = WordpieceEncoder(args)
        with Pool(args.workers, initializer=encoder.initializer) as pool:
            encoded_lines = pool.imap(encoder.encode_lines, zip(*input_files), 100)

            stats = Counter()
            for i, (filt, enc_lines) in enumerate(encoded_lines, start=1):
                if filt == "PASS":
                    for enc_line, output_h in zip(enc_lines, output_files):
                        print(enc_line, file=output_h)
                else:
                    stats["num_filtered_" + filt] += 1
                if i % 10000 == 0:
                    print("processed {} lines".format(i), file=sys.stderr)

            for k, v in stats.most_common():
                print("[{}] filtered {} lines".format(k, v), file=sys.stderr)
