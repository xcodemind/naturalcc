# -*- coding: utf-8 -*-
import os
# *refrenence: https://github.com/github/CodeSearchNet/blob/master/notebooks/ExploreData.ipynb
import itertools
from glob import glob
import pandas as pd
from typing import Dict
from dataset.codesearchnet.utils import constants
from ncc.utils.utils import mkdir
import argparse

def download(data_dir: str) -> None:
    # step 1: download original data from
    #          https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{python,java,go,php,ruby,javascript}.zip
    mkdir(data_dir)

    for lang in constants.LANGUAGES:
        data_file = os.path.join(data_dir, '{}.zip'.format(lang))
        if os.path.exists(data_file):
            print('file {} exists, skip this one.'.format(data_file))
        else:
            os.system('wget -P  https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{}}.zip'.format(lang))


def preprocess(data_dir: str) -> Dict:
    # step 2: unzip *.zip and futhermore, unzip its train/valid/test.zip dataset
    gz_files = {}
    for lang in constants.LANGUAGES:
        zip_file = os.path.join(data_dir, '{}.zip'.format(lang))
        if os.path.exists(os.path.join(data_dir, lang)):
            # if data_dir has already been unzipped, skip *.zip file
            pass
        else:
            os.system('unzip -d {} {}'.format(data_dir, zip_file))

        gz_files[lang] = []
        for mode in constants.MODES:
            data_folder = os.path.join(data_dir, lang, 'final/jsonl', mode)
            for gz_file in glob('{}/*.jsonl.gz'.format(data_folder)):
                gz_files[lang].append(gz_file)
        gz_files[lang] = sorted(gz_files[lang])
    return gz_files


def statistic(gz_files, vis=False):
    # step 3: load gz
    COLUMNS = ['repo', 'path', 'url', 'code', 'code_tokens', 'docstring',
               'docstring_tokens', 'language', 'partition']

    # columns_short_list = ['code_tokens', 'docstring_tokens', 'language', 'partition']

    def jsonl_list_to_dataframe(file_list, columns=COLUMNS):
        """Load a list of jsonl.gz files into a pandas DataFrame."""
        return pd.concat([pd.read_json(f, orient='records', compression='gzip',
                                       lines=True)[columns]
                          for f in file_list], sort=False)

    all_files = sorted(list(itertools.chain(*gz_files.values())))  # all languages
    # all_files = sorted(gz_files['ruby'])  # ruby only, for test
    all_df = jsonl_list_to_dataframe(all_files)
    for (lang, mode), index in all_df.groupby(['language', 'partition', ]).groups.items():
        print('{} {}: {}'.format(lang, mode, index.size))

    if vis:
        # plot()
        pass

def explore(gz_files: Dict) -> None:
    # 1. data statistics
    statistic(gz_files)

    # 2. show bad examples


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
    parser = argparse.ArgumentParser()
    # Before creating the true parser, we need to import optional user module
    # in order to eagerly import custom tasks, optimizers, architectures, etc.
    parser.add_argument('--data_dir', default='/data/wanyao/ghproj_d/naturalcodev3/codesearchnet/raw')
    args_ = parser.parse_args()
    # data_dir = '/data/wanyao/ghproj_d/CodeSearchNet/data'
    download(args_.data_dir)
    gz_files = preprocess(args_.data_dir)
    explore(gz_files)
