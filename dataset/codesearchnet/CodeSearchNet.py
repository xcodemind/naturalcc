# -*- coding: utf-8 -*-

from typing import *

import os
import wget
import zipfile
import glob
import gzip
import shutil
import numpy as np
import json, ujson
from tree_sitter import Language

from dataset.codesearchnet.utils import constants
from dataset.codesearchnet.utils import util
from dataset.codesearchnet.parser import CodeParser

from ncc.multiprocessing import mpool


class CodeSearchNet(object):
    """
    Thanks to Tree-Sitter team.
    Tools: https://github.com/tree-sitter/py-tree-sitter
    """

    resources = {
        'ruby': 'https://codeload.github.com/tree-sitter/tree-sitter-ruby/zip/master',
        'go': 'https://codeload.github.com/tree-sitter/tree-sitter-go/zip/master',
        'javascript': 'https://codeload.github.com/tree-sitter/tree-sitter-javascript/zip/master',
        'python': 'https://codeload.github.com/tree-sitter/tree-sitter-python/zip/master',
        'php': 'https://codeload.github.com/tree-sitter/tree-sitter-php/zip/master',
        'java': 'https://codeload.github.com/tree-sitter/tree-sitter-java/zip/master',
    }

    attrs = set(['code', 'code_tokens', 'docstring', 'docstring_tokens', 'func_name', 'original_string',
                 'path', 'repo', 'sha', 'url'])

    _LIB_DIR = 'libs'  # tree-sitter libraries
    _SO_DIR = 'so'  # tree-sitter *.so files
    _RAW_DIR = 'raw'  # raw data
    _DATA_DIR = 'data'  # flatten data

    def __init__(self, root: str = None, download: bool = False,
                 thread_num: int = None):
        if root is None:
            root = '~/.ncc/{}'.format(self.__class__.__name__)
        self.root = os.path.expanduser(root)
        os.makedirs(self.root, exist_ok=True)
        self._update_dirs()

        # download libs/*.zip or data/*.zip
        if download:
            for lng, url in self.resources.items():
                self._download(lng, url)

        # check libs/*.zip or data/*.zip exist
        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        # build *.so from libs/*.zip and extract files from data/*.zip
        for lng in self.resources.keys():
            self.build(lng)

        self.pool = mpool.MPool(thread_num)
        self._set_raw_files()

    def _update_dirs(self):
        self._LIB_DIR = os.path.join(self.root, self._LIB_DIR)
        self._SO_DIR = os.path.join(self.root, self._SO_DIR)
        self._RAW_DIR = os.path.join(self.root, self._RAW_DIR)
        self._DATA_DIR = os.path.join(self.root, self._DATA_DIR)

    def _check_exists(self) -> bool:
        for lng in self.resources.keys():
            if os.path.exists(os.path.join(self._LIB_DIR, '{}.zip'.format(lng))) and \
                    os.path.exists(os.path.join(self._RAW_DIR, '{}.zip'.format(lng))):
                pass
            else:
                return False
        return True

    def _build_so(self, lng: str):
        # build so file for language
        os.makedirs(self._SO_DIR, exist_ok=True)
        so_file = os.path.join(self._SO_DIR, '{}.so'.format(lng))
        if not os.path.exists(so_file):
            lib_file = os.path.join(self._LIB_DIR, '{}.zip'.format(lng))
            with zipfile.ZipFile(lib_file, 'r') as zip_file:
                zip_file.extractall(path=self._LIB_DIR)

        Language.build_library(
            # your language parser file
            so_file,
            # Include one or more languages
            [os.path.join(self._LIB_DIR, 'tree-sitter-{}-master'.format(lng))],
        )

    def _extract_data(self, lng: str):
        data_file = os.path.join(self._DATA_DIR, '{}.zip'.format(lng))
        src_dst_files = []
        with zipfile.ZipFile(data_file, 'r') as file_list:
            for file_info in file_list.filelist:
                src_file = file_info.filename
                if str.endswith(src_file, '.jsonl.gz'):
                    # exist in current direction and their sizes are same
                    dst_file = os.path.join(self._DATA_DIR, lng, os.path.split(src_file)[-1])
                    if (os.path.exists(dst_file)) and (os.path.getsize(dst_file) == file_info.file_size):
                        pass
                    else:
                        file_list.extract(src_file, path=self._DATA_DIR)
                        src_file = os.path.join(self._DATA_DIR, src_file)
                        src_dst_files.append([src_file, dst_file])
        for src_file, dst_file in src_dst_files:
            dst_file = os.path.join(self._DATA_DIR, lng, os.path.split(src_file)[-1])
            shutil.move(src=src_file, dst=dst_file)
        # delete tmp direction, e.g. ~/.ncc/data/ruby/final
        del_dir = os.path.join(self._DATA_DIR, lng, 'final')
        shutil.rmtree(del_dir, ignore_errors=True)

    def build(self, lng: str):
        self._build_so(lng)
        self._extract_data(lng)

    def update(self, lng: str, url=None):
        assert lng in self.resources, RuntimeError('{} is not in {}'.format(lng, list(self.resources.keys())))
        self._download(lng, url, exist_ignore=False)
        self.build(lng)

    def _download_lib(self, lng: str, url: str, exist_ignore=True):
        # download lib file from github.com
        os.makedirs(self._LIB_DIR, exist_ok=True)
        lib_file = os.path.join(self._LIB_DIR, '{}.zip'.format(lng))
        if os.path.exists(lib_file) and exist_ignore:
            # print('file({}) exists, ignore this.'.format(lib_file))
            pass
        else:
            wget.download(url=url, out=lib_file)

    def _download_data(self, lng: str, exist_ignore=True):
        """
        download original data from
        https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{python,java,go,php,ruby,javascript}.zip
        """
        os.makedirs(self._DATA_DIR, exist_ok=True)
        data_file = os.path.join(self._DATA_DIR, '{}.zip'.format(lng))
        if os.path.exists(data_file) and exist_ignore:
            # print('file({}) exists, ignore this.'.format(data_file))
            pass
        else:
            data_url = 'https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{}.zip'.format(lng)
            wget.download(url=data_url, out=data_file)

    def _download(self, lng: str, url: str, exist_ignore=True):
        self._download_lib(lng, url, exist_ignore)
        self._download_data(lng, exist_ignore)

    def _set_raw_files(self):
        self.raw_files = {
            lng: {
                mode: sorted(
                    glob.glob(os.path.join(self._DATA_DIR, lng, '*{}*.jsonl.gz'.format(mode))),
                    key=lambda filename: int(os.path.split(filename)[-1].split('.')[0].split('_')[-1]),
                )
                for mode in constants.MODES
            }
            for lng in constants.LANGUAGES
        }

    def _get_raw_file_len(self) -> Dict:
        raw_file_len = {}
        for lng in constants.LANGUAGES:
            raw_file_len[lng] = {}
            for mode in constants.MODES:
                raw_file_len[lng][mode] = self.pool.feed(
                    func=util.raw_data_len,
                    params=[(file,) for file in self.raw_files[lng][mode]]
                )
        return raw_file_len

    def _get_start_idx(self) -> Dict:
        """add index for reach raw file"""
        raw_file_lens = self._get_raw_file_len()
        raw_start_idx = {
            lng: {
                mode: [0] + np.cumsum(raw_file_lens[lng][mode]).tolist()[:-1]
                for mode in constants.MODES
            }
            for lng in constants.LANGUAGES
        }
        return raw_start_idx

    def _flatten(self, lng: str, save_attrs: List, raw_file: str, mode: str,
                 start_idx: int = None):
        if start_idx is not None:
            save_attrs.append('index')

        so_file = os.path.join(self._SO_DIR, '{}.so'.format(lng))
        parser = CodeParser(so_file, lng)
        reader = gzip.GzipFile(raw_file, 'r')
        writers = {}
        for attr in save_attrs:
            writer_dir = os.path.join(self._DATA_DIR, lng, attr)
            os.makedirs(writer_dir, exist_ok=True)
            writers[attr] = open(os.path.join(writer_dir, os.path.split(raw_file)[-1]), 'w')
        MAX_SUB_TOKEN_LEN = 0

        data_line = reader.readline().strip()
        while len(data_line) > 0:
            code_info = json.loads(data_line)
            code_info['index'] = start_idx  # add index for data

            for _, node in data_line['raw_ast'].items():
                if len(node['children']) == 1:
                    max_sub_token_len = max(max_sub_token_len, len(util.split_identifier(node['children'][0])))
            MAX_SUB_TOKEN_LEN = max(MAX_SUB_TOKEN_LEN, max_sub_token_len)

            for key, entry in data_line.items():
                writers[key].write(ujson.dumps(entry) + '\n')

            start_idx += 1
            data_line = reader.readline().strip()

    def flatten_data(self, lng: str, save_attrs=List):
        def _check_attrs_exist() -> List:
            excluded_attrs = []
            for attr in save_attrs:
                if attr not in self.attrs:
                    excluded_attrs.append(attr)
            return excluded_attrs

        excluded_attrs = _check_attrs_exist()
        if len(excluded_attrs) > 0:
            raise RuntimeError('{} Dataset do not include attributes:{}.'.format(
                self.__class__.__name__, excluded_attrs))

        start_idx = self._get_start_idx()
        params = []
        for mode in constants.MODES:
            for data_file, id in zip(self.raw_files[lng][mode], start_idx[mode]):
                params.append((lng, save_attrs, data_file, mode, id,))
        self.pool.feed(func=self._flatten, params=params, )


if __name__ == '__main__':
    dataset = CodeSearchNet()
    result = dataset.get_raw_file_len()
    print(result)
