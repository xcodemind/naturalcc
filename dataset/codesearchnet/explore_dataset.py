# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.path.abspath('.'))

# *refrenence: https://github.com/github/CodeSearchNet/blob/master/notebooks/ExploreData.ipynb

import itertools
from glob import glob
import pandas as pd

from typing import *

from dataset.utils.constants import LANGUAGES, MODES
from src.utils.util import mkdir


def download(data_dir: str) -> None:
    # step 1: download original data from
    #          https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{python,java,go,php,ruby,javascript}.zip
    mkdir(data_dir)

    for lang in LANGUAGES:
        data_file = os.path.join(data_dir, '{}.zip'.format(lang))
        if os.path.exists(data_file):
            print('file {} exists, skip this one.'.format(data_file))
        else:
            # pass
            os.system('wget -P  https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{}}.zip'.format(lang))


def preprocess(data_dir: str) -> Dict:
    # step 2: unzip *.zip and futhermore, unzip its train/valid/test.zip dataset
    gz_files = {}
    for lang in LANGUAGES:
        data_file = os.path.join(data_dir, '{}.zip'.format(lang))
        if os.path.exists(os.path.join(data_dir, lang)):
            # if data_dir has already been unzipped, skip *.zip file
            pass
        else:
            os.system('unzip -d {} {}'.format(data_dir, data_file))

        gz_files[lang] = []
        for md in MODES:
            data_file_md = os.path.join(data_dir, lang, 'final/jsonl', md)
            for md_file in glob('{}/*.jsonl.gz'.format(data_file_md)):
                gz_files[lang].append(md_file)
        gz_files[lang] = sorted(gz_files[lang])
    return gz_files


def explore(gz_files: Dict) -> None:
    # step 3: load gz
    columns_long_list = ['repo', 'path', 'url', 'code', 'code_tokens', 'docstring',
                         'docstring_tokens', 'language', 'partition']

    # columns_short_list = ['code_tokens', 'docstring_tokens', 'language', 'partition']

    def jsonl_list_to_dataframe(file_list, columns=columns_long_list):
        """Load a list of jsonl.gz files into a pandas DataFrame."""
        return pd.concat([pd.read_json(f, orient='records', compression='gzip',
                                       lines=True)[columns]
                          for f in file_list], sort=False)

    all_files = sorted(list(itertools.chain(*gz_files.values())))  # all languages
    # all_files = sorted(gz_files['ruby'])  # ruby only, for test
    all_df = jsonl_list_to_dataframe(all_files)
    for (lang, md), index in all_df.groupby(['language', 'partition', ]).groups.items():
        print('{} {}: {}'.format(lang, md, index.size))


if __name__ == '__main__':
    '''
    language    partition
    go test: 14291
    go train: 317832
    go valid: 14242
    java test: 26909
    java train: 454451
    java valid: 15328
    javascript test: 6483
    javascript train: 123889
    javascript valid: 8253
    php test: 28391
    php train: 523712
    php valid: 26015
    python test: 22176
    python train: 412178
    python valid: 23107
    ruby test: 2279
    ruby train: 48791
    ruby valid: 2209
    '''

    data_dir = '/data/wanyao/ghproj_d/CodeSearchNet/data'
    download(data_dir)
    gz_files = preprocess(data_dir)
    explore(gz_files)
