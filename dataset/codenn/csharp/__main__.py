# -*- coding: utf-8 -*-

import os
import re
import wget
import ujson
import itertools
from ncc.data.constants import MODES
from dataset.codenn.csharp.py2x import (parse_csharp_code, parse_csharp_docstring)
from multiprocessing import Pool, cpu_count
from ncc import LOGGER

LANGUAGE = 'csharp'
_CODE_NN_ROOT_DIR = os.path.expanduser('~/.ncc/codenn')
RAW_DIR = os.path.join(_CODE_NN_ROOT_DIR, 'raw', LANGUAGE)
os.makedirs(RAW_DIR, exist_ok=True)
FLATTEN_DIR = os.path.join(_CODE_NN_ROOT_DIR, 'flatten', LANGUAGE)
os.makedirs(FLATTEN_DIR, exist_ok=True)
LIB_DIR = os.path.join(_CODE_NN_ROOT_DIR, 'libs', LANGUAGE)
os.makedirs(LIB_DIR, exist_ok=True)


def download():
    raw_data = [
        'https://raw.githubusercontent.com/sriniiyer/codenn/master/data/stackoverflow/csharp/train.txt',
        'https://raw.githubusercontent.com/sriniiyer/codenn/master/data/stackoverflow/csharp/valid.txt',
        'https://raw.githubusercontent.com/sriniiyer/codenn/master/data/stackoverflow/csharp/test.txt',
    ]
    for url in raw_data:
        out = os.path.join(RAW_DIR, os.path.split(url)[-1])
        if not os.path.exists(out):
            wget.download(url, out)


def build_so():
    import zipfile
    tree_sitter_url = 'https://github.com/tree-sitter/tree-sitter-c-sharp/archive/master.zip'
    lib_filename = os.path.join(LIB_DIR, 'tree-sitter-c-sharp-master.zip')
    if not os.path.exists(lib_filename):
        wget.download(tree_sitter_url, lib_filename)
    dir_filename = lib_filename[:str.rfind(lib_filename, '.zip')]
    if not os.path.exists(dir_filename):
        with zipfile.ZipFile(lib_filename, 'r') as zip_file:
            zip_file.extractall(path=LIB_DIR)

    from tree_sitter import Language

    so_file = os.path.join(LIB_DIR, '{}.so'.format(LANGUAGE))
    if not os.path.exists(so_file):
        Language.build_library(
            # your language parser file, we recommend buidl *.so file for each language
            so_file,
            # Include one or more languages
            [os.path.join(LIB_DIR, 'tree-sitter-c-sharp-master')],
        )


def flatten_data():
    """
    raw data:
        6602 test.txt
       52812 train.txt
        6601 valid.txt

    flatten data:
        6599 test.code
        6599 test.docstring
       52795 train.code
       52795 train.docstring
        6599 valid.code
        6599 valid.docstring
    """
    for mode in MODES:
        raw_filename = os.path.join(RAW_DIR, '{}.txt'.format(mode))
        code_filename = os.path.join(FLATTEN_DIR, '{}.code'.format(mode))
        docstring_filename = os.path.join(FLATTEN_DIR, '{}.docstring'.format(mode))
        LOGGER.info(
            'Flatten {} into code({}) and docstring({}).'.format(raw_filename, code_filename, docstring_filename))
        with open(raw_filename, 'r') as reader, \
            open(code_filename, 'w') as code_writer, open(docstring_filename, 'w') as docstring_writer:
            for idx, line in enumerate(reader):
                """example: [\d+]\t[\d+]\t[docstring]\t[code]\t0\n"""
                try:
                    parsed_line = line.rstrip('\n').split('\t')
                    assert len(parsed_line) == 5, AssertionError(idx, line)
                except AssertionError:
                    continue
                docstring, code = parsed_line[2].strip(), parsed_line[3].strip()
                docstring, code = map(lambda string: string.replace('\\r\\n', '\n').replace('\\n', '\n'),
                                      (docstring, code,))
                print(ujson.dumps(docstring, ensure_ascii=False), file=docstring_writer)
                print(ujson.dumps(code, ensure_ascii=False), file=code_writer)


def find_offsets(filename, num_chunks):
    with open(filename, "r", encoding="utf-8") as f:
        size = os.fstat(f.fileno()).st_size
        chunk_size = size // num_chunks
        offsets = [0 for _ in range(num_chunks + 1)]
        for i in range(1, num_chunks):
            f.seek(chunk_size * i)
            safe_readline(f)
            offsets[i] = f.tell()
        return offsets


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


def get_tokenizer(modality):
    if modality == 'code':
        return parse_csharp_code
    elif modality == 'docstring':
        return parse_csharp_docstring
    else:
        raise NotImplementedError


def parse_fn(raw_filename, dst_filename, modality, start=0, end=-1):
    token_fn = get_tokenizer(modality)
    with open(raw_filename, 'r', encoding='UTF-8') as reader, open(dst_filename, 'w') as writer:
        reader.seek(start)
        line = safe_readline(reader)
        while line:
            if end > 0 and reader.tell() > end:
                break
            line = ujson.loads(line)
            tokens = token_fn(line)
            print(ujson.dumps(tokens), file=writer)
            line = safe_readline(reader)


def tokenization():
    def _cat(src_filenames, tgt_filename):
        cmd = 'cat {} > {}'.format(' '.join(src_filenames), tgt_filename)
        # run cat
        os.system(cmd)

    modalities = ['docstring', 'code', ]
    num_workers = cpu_count()
    # num_workers = 10

    for mode, modality in itertools.product(MODES, modalities):
        raw_filename = os.path.join(FLATTEN_DIR, '{}.{}'.format(mode, modality))
        dst_filename = raw_filename + '_tokens'
        offsets = find_offsets(raw_filename, num_workers)

        with Pool(num_workers) as mpool:
            result = [
                mpool.apply_async(
                    parse_fn,
                    (raw_filename, dst_filename + str(idx), modality, offsets[idx], offsets[idx + 1])
                )
                for idx in range(num_workers)
            ]
            result = [res.get() for res in result]

        _cat([dst_filename + str(idx) for idx in range(num_workers)], dst_filename)
        for idx in range(num_workers):
            os.remove(dst_filename + str(idx))


if __name__ == '__main__':
    """
    nohup python -m dataset.codenn.csharp.__main__ > codenn.log 2>&1 &
    """
    download()
    build_so()
    flatten_data()
    tokenization()
