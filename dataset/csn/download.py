# -*- coding: utf-8 -*-


import os
import wget
import argparse
import zipfile
import shutil
import gdown
from tree_sitter import Language

try:
    from dataset.csn import (
        LANGUAGES, RAW_DATA_DIR, LIBS_DIR,
        LOGGER,
    )
except ImportError:
    from . import (
        LANGUAGES, RAW_DATA_DIR, LIBS_DIR,
        LOGGER,
    )


def download_file(url, local):
    """download raw data files from amazon and lib files from github.com"""
    _local = os.path.expanduser(local)
    os.makedirs(os.path.dirname(_local), exist_ok=True)
    if os.path.exists(_local):
        LOGGER.info('File {} exists, ignore it. If you want to overwrite it, pls delete it firstly.'.format(local))
    else:
        LOGGER.info('Download {} from {}'.format(local, url))
        wget.download(url=url, out=_local)


def gdownload_file(url, local):
    """download raw data files from amazon and lib files from github.com"""
    _local = os.path.expanduser(local)
    os.makedirs(os.path.dirname(_local), exist_ok=True)
    if os.path.exists(_local):
        LOGGER.info('File {} exists, ignore it. If you want to overwrite it, pls delete it firstly.'.format(local))
    else:
        LOGGER.info('Download {} from {}'.format(local, url))
        # wget.download(url=url, out=_local)
        gdown.download(url, _local, quiet=False)


def unzip_raw_data(raw_dir, lang):
    """unzip raw data from ~/.ncc/code_search_net/raw to ~/.ncc/code_search_net/raw_unzip"""
    _raw_dir = os.path.expanduser(raw_dir)
    LOGGER.info('Extracting raw data({})'.format(raw_dir))
    src_dst_files = []
    raw_data_path = os.path.join(_raw_dir, '{}.zip'.format(lang))
    with zipfile.ZipFile(raw_data_path, 'r') as file_list:
        for file_info in file_list.filelist:
            src_file = file_info.filename
            if str.endswith(src_file, '.jsonl.gz'):
                # temporarily decompress data at {tmp_dst_file}
                tmp_dst_file = os.path.join(_raw_dir, lang, os.path.split(src_file)[-1])
                os.makedirs(os.path.dirname(tmp_dst_file), exist_ok=True)
                if not os.path.exists(tmp_dst_file):
                    file_list.extract(src_file, path=_raw_dir)
                    src_file = os.path.join(_raw_dir, src_file)
                    src_dst_files.append([src_file, tmp_dst_file])
    for src_file, dst_file in src_dst_files:
        dst_file = os.path.join(_raw_dir, lang, os.path.split(src_file)[-1])
        shutil.move(src=src_file, dst=dst_file)
    # delete temporary directory, e.g. mv ~/.ncc/data/ruby/final/*.jsonl.gz to ~/.ncc/data/ruby/*.jsonl.gz
    del_dir = os.path.join(_raw_dir, lang, 'final')
    shutil.rmtree(del_dir, ignore_errors=True)


def build_so(lib_dir, lang):
    """build so file for certain language with Tree-Sitter"""
    _lib_dir = os.path.expanduser(lib_dir)
    lib_file, _lib_file = os.path.join(lib_dir, '{}.zip'.format(lang)), os.path.join(_lib_dir, '{}.zip'.format(lang))
    if os.path.exists(_lib_file):
        LOGGER.info('Tree-Sitter so file for {} does not exists, compiling.'.format(lib_file))
        # decompress Tree-Sitter library
        with zipfile.ZipFile(_lib_file, 'r') as zip_file:
            zip_file.extractall(path=_lib_dir)
        so_file, _so_file = os.path.join(lib_dir, '{}.so'.format(lang)), os.path.join(_lib_dir, '{}.so'.format(lang))
        LOGGER.info('Building Tree-Sitter compile file {}'.format(so_file))
        Language.build_library(
            # your language parser file, we recommend buidl *.so file for each language
            _so_file,
            # Include one or more languages
            [os.path.join(_lib_dir, 'tree-sitter-{}-master'.format(lang))],
        )
    else:
        LOGGER.info('Tree-Sitter so file for {} exists, ignore it.'.format(lib_file))


if __name__ == '__main__':
    """
    This script is
        1) to download one language dataset from CodeSearchNet and the corresponding library
        2) to decompress raw data file and Tree-Sitter libraries
        3) to compile Tree-Sitter libraries into *.so file

    # ====================================== CodeSearchNet ====================================== #
    Dataset: raw CodeSearchNet data files of Java/Javascript/PHP/GO/Ruby/Python
       # language   # URL                                                                       # size
       java:        https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip			1060569153
       javascript:  https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/javascript.zip	1664713350
       php:         https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/php.zip			851894048
       go:          https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/go.zip			487525935
       ruby:        https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/ruby.zip			111758028
       python:      https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip		940909997
       
    Tree-Sitter libs has been updated and thus they cannot match this repo.. 
    Therefore, we recommend use our built so files, or you can build it with old version libs(stored on google driver).
    
    Tree-Sitter: AST generation tools, TreeSitter repositories from Github can be updated, therefore their size is capricious
       # language   # URL
       Java:        https://codeload.github.com/tree-sitter/tree-sitter-java/zip/master
       Javascript:  https://codeload.github.com/tree-sitter/tree-sitter-javascript/zip/master
       PHP:         https://codeload.github.com/tree-sitter/tree-sitter-php/zip/master
       GO:          https://codeload.github.com/tree-sitter/tree-sitter-go/zip/master
       Ruby:        https://codeload.github.com/tree-sitter/tree-sitter-ruby/zip/master
       Python:      https://codeload.github.com/tree-sitter/tree-sitter-python/zip/master
       # language   # google driver
       csharp:      https://drive.google.com/uc?id=1nbUVW7WRWH-4B5JRT-Q_nJVPQ9sAfhwW,
       go:          https://drive.google.com/uc?id=13cmMblgg1FeEIqk7YJlMUU59WM3y16en,
       java:        https://drive.google.com/uc?id=1JIvpp3FVjrA1MVasin1RnFNy3-JQoKb0,
       javascript:  https://drive.google.com/uc?id=13f7XqmIiXU633NLmTf3aqNEKrtQFNT2I,
       php:         https://drive.google.com/uc?id=1uUtKmZ_lu9K-6W610Lx-VDme6QLiItum,
       python:      https://drive.google.com/uc?id=1Z6R7VzZYwCCpu4ZK8ir0oBMeN9DYnepm,
       ruby:        https://drive.google.com/uc?id=1v-evhEXR2Hff9melWSUJSfou9J1SRCaM,   
    """
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--language", "-l", default=LANGUAGES, type=str, nargs='+', help="languages constain [{}]".format(LANGUAGES),
    )
    parser.add_argument(
        "--dataset_dir", "-d", default=RAW_DATA_DIR, type=str, help="raw dataset download directory",
    )
    parser.add_argument(
        "--libs_dir", "-b", default=LIBS_DIR, type=str, help="tree-sitter library directory",
    )
    args = parser.parse_args()
    # print(args)

    # g_urls = {
    #     'csharp': 'https://drive.google.com/uc?id=1nbUVW7WRWH-4B5JRT-Q_nJVPQ9sAfhwW',
    #     'go': 'https://drive.google.com/uc?id=13cmMblgg1FeEIqk7YJlMUU59WM3y16en',
    #     'java': 'https://drive.google.com/uc?id=1JIvpp3FVjrA1MVasin1RnFNy3-JQoKb0',
    #     'javascript': 'https://drive.google.com/uc?id=13f7XqmIiXU633NLmTf3aqNEKrtQFNT2I',
    #     'php': 'https://drive.google.com/uc?id=1uUtKmZ_lu9K-6W610Lx-VDme6QLiItum',
    #     'python': 'https://drive.google.com/uc?id=1Z6R7VzZYwCCpu4ZK8ir0oBMeN9DYnepm',
    #     'ruby': 'https://drive.google.com/uc?id=1v-evhEXR2Hff9melWSUJSfou9J1SRCaM',
    # }

    for lang in args.language:
        # download raw data from amazon CSN datasets
        dataset_url = 'https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{}.zip'.format(lang)
        dataset_filename = os.path.join(args.dataset_dir, '{}.zip'.format(lang))
        download_file(url=dataset_url, local=dataset_filename)
        # decompress raw data
        unzip_raw_data(raw_dir=args.dataset_dir, lang=lang)

        # # download Tree-Sitter AST parser libraries
        # lib_url = 'https://drive.google.com/uc?id={}'.format(g_urls[lang])
        # lib_filename = os.path.join(args.libs_dir, '{}.zip'.format(lang))
        # gdownload_file(url=lib_url, local=lib_filename)
        # # compiling Tree-Sitter so file
        # build_so(lib_dir=args.libs_dir, lang=lang)
