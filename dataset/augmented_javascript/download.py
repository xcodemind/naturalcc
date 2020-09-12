# -*- coding: utf-8 -*-

import os
import wget
import argparse


try:
    from dataset.augmented_javascript import (
        LANGUAGES, RAW_DATA_DIR, RAW_DATA_DIR_TYPE_PREDICTION,
        LOGGER,
    )
except ImportError:
    from . import (
        LANGUAGES, RAW_DATA_DIR,
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
        LOGGER.info('Finished downloading {}'.format(url))


if __name__ == '__main__':
    '''
    Downloading augmented_javascript dataset(s)
    ref: https://github.com/parasj/contracode
    '''
    parser = argparse.ArgumentParser(
        description="Downloading augmented_javascript dataset(s)")
    parser.add_argument(
        "--dataset_dir", "-d", default=RAW_DATA_DIR, type=str, help="raw dataset download directory",
    )
    parser.add_argument(
        "--dataset_dir_type_prediction", "-d", default=RAW_DATA_DIR_TYPE_PREDICTION, type=str, help="raw dataset download directory",
    )
    args = parser.parse_args()

    # 1. download: javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz
    dataset_filename = os.path.join(args.dataset_dir, 'javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz')
    download_file(url='https://drive.google.com/file/d/1YfHvacsAn9ngfjiJYbdo8LiFUqkbroxj/view?usp=sharing', local=dataset_filename)

    # 1. download: javascript_augmented.pickle.gz
    dataset_filename = os.path.join(args.dataset_dir, 'javascript_augmented.pickle.gz')
    download_file(url='https://drive.google.com/file/d/1YfPTPPOv4evldpN-n_4QBDWDWFImv7xO/view?usp=sharing',
                  local=dataset_filename)

    # 2. Unzip: Given a .gz file, unzip it. TODO: unzip_raw_data 有点绕

    # 3. dowload `types` for downstream task - type prediction
    dataset_filename = os.path.join(args.dataset_dir_type_prediction, 'target_wl')
    download_file(url='https://drive.google.com/file/d/1YtLVoMUsxU6HTpu5Qvs0xldm_SC_FZRz/view?usp=sharing',
                  local=dataset_filename)
    dataset_filename = os.path.join(args.dataset_dir_type_prediction, 'test_nounk.txt')
    download_file(url='https://drive.google.com/file/d/1YvoM6rxcaX1wsyQu0HbGurQdaV6fsmym/view?usp=sharing',
                  local=dataset_filename)
    dataset_filename = os.path.join(args.dataset_dir_type_prediction, 'test_projects_gold_filtered.json')
    download_file(url='https://drive.google.com/file/d/1YsoSKGhuOUw3CNAAzZ3j9Fm5C8itc8Jt/view?usp=sharing',
                  local=dataset_filename)
    dataset_filename = os.path.join(args.dataset_dir_type_prediction, 'train_nounk.txt')
    download_file(url='https://drive.google.com/file/d/1YunIabuWqd3V9kZssloXrOUvyfLcM7FH/view?usp=sharing',
                  local=dataset_filename)
    dataset_filename = os.path.join(args.dataset_dir_type_prediction, 'valid_nounk.txt')
    download_file(url='https://drive.google.com/file/d/1YvN5UQuijRgUAL3aF1MzGLn1NiVMTAEo/view?usp=sharing',
                  local=dataset_filename)

