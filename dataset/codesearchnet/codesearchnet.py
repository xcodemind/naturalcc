# -*- coding: utf-8 -*-

from typing import (
    List, Dict, Optional, Union
)

import os
import wget
import zipfile
import glob
import gzip
import shutil
import numpy as np
import json
import ujson
from copy import deepcopy
from tree_sitter import Language
from tqdm import tqdm

from dataset.codesearchnet.utils import constants
from dataset.codesearchnet.utils import util
from dataset.codesearchnet.utils import util_path
from dataset.codesearchnet.utils import util_ast
from dataset.codesearchnet.parser import CodeParser

# from ncc.multiprocessing import mpool
from ncc.multiprocessing import ppool
from ncc import LOGGER


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
                 'path', 'repo', 'sha', 'url', 'partition', 'language'])
    extended_tree_modalities = set(['path', 'sbt', 'sbtao', 'bin_ast'])

    _LIB_DIR = 'libs'  # tree-sitter libraries
    _SO_DIR = 'so'  # tree-sitter *.so files
    _RAW_DIR = 'raw'  # raw data
    _DATA_DIR = 'data'  # extracted data
    _FLATTEN_DIR = 'flatten'  # flatten data
    _MAX_SUB_TOKEN_LEN_FILENAME = 'max_sub_token_len.txt'

    def __init__(self, root: str = None, download: bool = False,
                 thread_num: int = None):
        if root is None:
            root = '~/.ncc/{}'.format(self.__class__.__name__)
        self.root = os.path.expanduser(root)
        os.makedirs(self.root, exist_ok=True)
        self._update_dirs()

        # download libs/*.zip or raw/*.zip
        if download:
            for lng, url in self.resources.items():
                self._download(lng, url)

        # check libs/*.zip or raw/*.zip exist
        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        # build *.so from libs/*.zip and extract files from raw/*.zip
        for lng in self.resources.keys():
            self.build(lng)

        self.pool = ppool.PPool(thread_num)
        self._set_raw_files()

    def _update_dirs(self):
        self._LIB_DIR = os.path.join(self.root, self._LIB_DIR)
        self._SO_DIR = os.path.join(self.root, self._SO_DIR)
        self._RAW_DIR = os.path.join(self.root, self._RAW_DIR)
        self._DATA_DIR = os.path.join(self.root, self._DATA_DIR)
        self._FLATTEN_DIR = os.path.join(self.root, self._FLATTEN_DIR)

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

            LOGGER.info('Building Tree-Sitter compile file {}'.format(so_file))
            Language.build_library(
                # your language parser file
                so_file,
                # Include one or more languages
                [os.path.join(self._LIB_DIR, 'tree-sitter-{}-master'.format(lng))],
            )

    def _extract_data(self, lng: str):
        data_file = os.path.join(self._RAW_DIR, '{}.zip'.format(lng))
        src_dst_files = []
        LOGGER.info('Extracting raw data files from {} to {}. If already existed, skip it.'.format(
            data_file, self._DATA_DIR))
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
        """update(overwrite) data and compile files"""
        assert lng in self.resources, RuntimeError('{} is not in {}'.format(lng, list(self.resources.keys())))
        self._download(lng, url, overwrite=False)
        self.build(lng)

    def _download_lib(self, lng: str, url: str, overwrite=True):
        # download lib file from github.com
        os.makedirs(self._LIB_DIR, exist_ok=True)
        lib_file = os.path.join(self._LIB_DIR, '{}.zip'.format(lng))
        if os.path.exists(lib_file) and overwrite:
            # print('file({}) exists, ignore this.'.format(lib_file))
            pass
        else:
            LOGGER.info('Download Tree-Sitter Library {} from {}'.format(lib_file, url))
            wget.download(url=url, out=lib_file)

    def _download_raw_data(self, lng: str, overwrite=True):
        """
        download original data from
        https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{python,java,go,php,ruby,javascript}.zip
        """
        os.makedirs(self._RAW_DIR, exist_ok=True)
        data_file = os.path.join(self._RAW_DIR, '{}.zip'.format(lng))
        if os.path.exists(data_file) and overwrite:
            # print('file({}) exists, ignore this.'.format(data_file))
            pass
        else:
            data_url = 'https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{}.zip'.format(lng)
            LOGGER.info('Download CodeSearchNet dataset {} from {}'.format(data_file, data_url))
            wget.download(url=data_url, out=data_file)

    def _download(self, lng: str, url: str, overwrite=True):
        self._download_lib(lng, url, overwrite)
        self._download_raw_data(lng, overwrite)

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

    def _get_raw_file_len(self, lng: str) -> Dict:
        raw_file_len = {}
        assert lng in constants.LANGUAGES, RuntimeError('{} doesn\'t exist in {}'.format(lng, constants.LANGUAGES))
        raw_file_len[lng] = {}
        for mode in constants.MODES:
            raw_file_len[lng][mode] = self.pool.feed(
                func=util.raw_data_len,
                params=[(file,) for file in self.raw_files[lng][mode]]
            )
        return raw_file_len

    def _get_start_idx(self, lng: str) -> Dict:
        """add index for reach raw file"""
        # raw_file_lens = {'ruby': {'train': [30000, 18791], 'valid': [2209], 'test': [2279]}}
        raw_file_lens = self._get_raw_file_len(lng)
        raw_start_idx = {
            mode: [0] + np.cumsum(raw_file_lens[lng][mode]).tolist()[:-1]
            for mode in constants.MODES
        }
        return raw_start_idx

    def _flatten(self, lng: str, save_attrs: List, raw_file: str, mode: str,
                 start_idx: int = None) -> int:

        def get_idx_filename(raw_filename) -> str:
            raw_filename = os.path.split(raw_filename)[-1]
            raw_file_idx = raw_filename.split('.')[0].split('_')[-1]
            idx_filename = '{}.txt'.format(raw_file_idx)
            return idx_filename

        if start_idx is not None:
            save_attrs.append('index')
        save_attrs = set(save_attrs)

        so_file = os.path.join(self._SO_DIR, '{}.so'.format(lng))
        parser = CodeParser(so_file, lng, to_lower=False)
        reader = gzip.GzipFile(raw_file, 'r')
        writers = {}
        for attr in save_attrs:
            writer_dir = os.path.join(self._FLATTEN_DIR, lng, mode, attr)
            os.makedirs(writer_dir, exist_ok=True)
            idx_file = get_idx_filename(raw_file)
            writers[attr] = open(os.path.join(writer_dir, idx_file), 'w')
        MAX_SUB_TOKEN_LEN = 0

        data_line = reader.readline().strip()
        while len(data_line) > 0:
            code_info = json.loads(data_line)
            # pop useless attributes
            for attr in self.attrs:
                if attr not in save_attrs:
                    code_info.pop(attr)
            code_info['index'] = start_idx  # add index for data
            if 'tok' in save_attrs:
                code_info['tok'] = parser.parse_code_tokens(code_info['code_tokens'])
                if code_info['tok'] is None:
                    code_info['tok'] = code_info['code_tokens']
            if 'raw_ast' in save_attrs:
                code_info['raw_ast'] = parser.parse_raw_ast(code_info['code'])
                # get max sub-tokens length for dgl training
                max_sub_token_len = 0
                if code_info['raw_ast'] is None:
                    pass
                else:
                    for _, node in code_info['raw_ast'].items():
                        if len(node['children']) == 1:
                            max_sub_token_len = max(max_sub_token_len, len(util.split_identifier(node['children'][0])))
                MAX_SUB_TOKEN_LEN = max(MAX_SUB_TOKEN_LEN, max_sub_token_len)
            if 'comment' in save_attrs:
                code_info['comment'] = parser.parse_comment(code_info['docstring'], code_info['docstring_tokens'])
                if code_info['comment'] is None:
                    code_info['comment'] = code_info['docstring_tokens']
            if 'method' in save_attrs:
                code_info['method'] = parser.parse_method(code_info['func_name'])
            # write
            for key, entry in code_info.items():
                writers[key].write(ujson.dumps(entry) + '\n')

            start_idx += 1
            data_line = reader.readline().strip()
        return MAX_SUB_TOKEN_LEN

    def flatten_data(self, lng: str, save_attrs: List = None, overwrite: bool = False):
        """flatten attributes separately"""

        def _check_attrs_valid() -> List:
            excluded_attrs = []
            for attr in save_attrs:
                if attr not in self.attrs:
                    excluded_attrs.append(attr)
            return excluded_attrs

        if save_attrs is None:
            save_attrs = ['code', 'code_tokens', 'docstring', 'docstring_tokens', 'func_name', 'original_string', ]
        else:
            # if save_attrs is set, then check
            excluded_attrs = _check_attrs_valid()
            if len(excluded_attrs) > 0:
                raise RuntimeError('{} Dataset do not include attributes:{}.'.format(
                    self.__class__.__name__, excluded_attrs))
        """
        1) [method] is sequence modal of [func_name]
        2) [raw_ast] is ast modal of [code]
        3) [tok] is refined from [code] or [code_tokens]
        4) [comment] is refined from [docstring] or [docstring_tokens]
        ** if [tok] or [comment] is None, use [code_tokens] or [docstring_tokens] as alternative
        """
        if 'func_name' in save_attrs:
            save_attrs.append('method')
        if 'code' in save_attrs:
            save_attrs.append('raw_ast')
        if ('code' in save_attrs) and ('code_tokens' in save_attrs):
            save_attrs.append('tok')
        if ('docstring' in save_attrs) and ('docstring_tokens' in save_attrs):
            save_attrs.append('comment')

        def _check_flatten_files_exist():
            """a simple check method"""
            for mode in constants.MODES:
                raw_file_num = len(self.raw_files[lng][mode])
                for attr in save_attrs:
                    tgt_dir = os.path.join(self._FLATTEN_DIR, lng, mode, attr)
                    if os.path.exists(tgt_dir) and len(os.listdir(tgt_dir)) == raw_file_num:
                        pass
                    else:
                        LOGGER.info('{}\'s flatten data don\'t exist, generating...'.format(lng))
                        return False
            return True

        if not overwrite and _check_flatten_files_exist():
            return

        start_idx = self._get_start_idx(lng)

        params = []
        for mode in constants.MODES:
            for data_file, id in zip(self.raw_files[lng][mode], start_idx[mode]):
                params.append((lng, save_attrs, data_file, mode, id,))
        # max_sub_token_len = self._flatten(*params[0])
        max_sub_token_len = self.pool.feed(func=self._flatten, params=params, )
        max_sub_token_len = max(max_sub_token_len)
        # write max_sub_token_len for bin-ast
        max_sub_token_len_filename = os.path.join(self._FLATTEN_DIR, lng, self._MAX_SUB_TOKEN_LEN_FILENAME)
        with open(max_sub_token_len_filename, 'w') as writer:
            writer.write(str(max_sub_token_len))
        LOGGER.info('Save flatten data and {}\'s max sub-token info in {}'.format(
            lng, os.path.join(self._FLATTEN_DIR, lng)))

    def flatten_data_all(self, save_attrs: List = None, overwrite: bool = False):
        for lng in constants.LANGUAGES:
            self.flatten_data(lng, save_attrs, overwrite)

    @staticmethod
    def _parse_new_tree_modalities(raw_ast_file: str, new_tree_modal_files: Dict, MAX_SUB_TOKEN_LEN: int, ):
        LOGGER.info('Start parsing {}'.format(new_tree_modal_files))
        reader = open(raw_ast_file, 'r')
        writers = {}
        for modal, modal_file in new_tree_modal_files.items():
            writers[modal] = open(modal_file, 'w')

        data_line = reader.readline().strip()
        while len(data_line) > 0:
            raw_ast = json.loads(data_line)
            # because some raw_ast it is too large and thus time-consuming, we ignore it
            if len(raw_ast) > constants.MAX_AST_NODE_NUM:
                raw_ast = None
            if raw_ast is None:
                # if no raw ast, its other tree modalities is None
                for writer in writers.values():
                    writer.write(ujson.dumps(raw_ast) + '\n')
            else:
                if 'path' in writers:
                    copy_raw_ast = deepcopy(raw_ast)
                    path = util_path.ast_to_path(copy_raw_ast, MAX_PATH=constants.MAX_AST_PATH_NUM)
                    writers['path'].write(ujson.dumps(path) + '\n')
                if ('sbt' in writers) or ('sbtao' in writers):
                    # sbt
                    copy_raw_ast = deepcopy(raw_ast)
                    padded_raw_ast = util_ast.pad_leaf_node(copy_raw_ast, MAX_SUB_TOKEN_LEN)
                    copy_padded_raw_ast = deepcopy(padded_raw_ast) if ('sbt' in writers) and ('sbtao' in writers) \
                        else padded_raw_ast
                    if 'sbt' in writers:
                        # write sbt
                        sbt = util_ast.parse_deepcom(padded_raw_ast, util_ast.build_sbt_tree, to_lower=False)
                        writers['sbt'].write(ujson.dumps(sbt) + '\n')
                    if 'sbtao' in writers:
                        # write sbtao
                        sbt = util_ast.parse_deepcom(copy_padded_raw_ast, util_ast.build_sbtao_tree, to_lower=False)
                        writers['sbtao'].write(ujson.dumps(sbt) + '\n')
                if 'bin_ast' in writers:
                    # ast
                    bin_ast = util_ast.parse_base(raw_ast)
                    writers['bin_ast'].write(ujson.dumps(bin_ast) + '\n')
            data_line = reader.readline().strip()
        reader.close()
        for writer in writers.values():
            writer.close()
        LOGGER.info('End parsing {}'.format(new_tree_modal_files))

    def parse_new_tree_modalities(self, lng: str, modalities: Optional[List] = None, overwrite: bool = False):

        def _check_modalities():
            excluded_tree_modalities = []
            for modal in modalities:
                if modal not in self.extended_tree_modalities:
                    excluded_tree_modalities.append(modal)
            if len(excluded_tree_modalities) > 0:
                raise RuntimeError('{} is not in our implement({})'.format(
                    excluded_tree_modalities, self.extended_tree_modalities))

        if modalities is None:
            modalities = list(self.extended_tree_modalities)
        else:
            _check_modalities()

        def _check_new_tree_modalities_exist():
            """a simple check method"""
            for mode in constants.MODES:
                raw_file_num = len(self.raw_files[lng][mode])
                for modal in modalities:
                    tgt_dir = os.path.join(self._FLATTEN_DIR, lng, mode, modal)
                    if os.path.exists(tgt_dir) and len(os.listdir(tgt_dir)) == raw_file_num:
                        pass
                    else:
                        LOGGER.info('{}\'s new tree modalities data don\'t exist, generating...'.format(lng))
                        return False
            return True

        if not overwrite and _check_new_tree_modalities_exist():
            return

        params = []
        # get language's max sub-token len from txt file
        max_sub_token_file = os.path.join(self._FLATTEN_DIR, lng, self._MAX_SUB_TOKEN_LEN_FILENAME)
        with open(max_sub_token_file, 'r') as reader:
            MAX_SUB_TOKEN_LEN = int(reader.read().strip())
        for mode in constants.MODES:
            raw_ast_dir = os.path.join(self._FLATTEN_DIR, lng, mode, 'raw_ast')
            assert os.path.exists(raw_ast_dir), \
                RuntimeError('{}\'s raw_ast dir {} doesn\'t exist, please generate first'.format(lng, raw_ast_dir))
            raw_ast_files = sorted(glob.glob(os.path.join(raw_ast_dir, '*.txt')))
            for file in raw_ast_files:
                new_tree_files = {}
                for modal in modalities:
                    modal_dir = os.path.join(self._FLATTEN_DIR, lng, mode, modal)
                    os.makedirs(modal_dir, exist_ok=True)
                    modal_file = os.path.join(modal_dir, os.path.split(file)[-1])
                    new_tree_files[modal] = modal_file
                params.append((file, new_tree_files, MAX_SUB_TOKEN_LEN,))
        # this may cause out of memory error because of recursion in ast processing, therefore we use single processor
        # self._parse_new_tree_modalities(*params[1])
        # for param in params:
        #     self._parse_new_tree_modalities(*param)
        self.pool.feed(func=CodeSearchNet._parse_new_tree_modalities, params=params, )

    def _lngs_init(self, lngs: Union[List, str, None] = None) -> List:
        if lngs is None:
            lngs = constants.LANGUAGES
        elif isinstance(lngs, str):
            lngs = [lngs]
        elif isinstance(lngs, list):
            pass
        else:
            raise NotImplementedError('Only None/List/str is available.')
        return lngs

    def parse_new_tree_modalities_all(self, lngs: Union[List, str, None] = None, modalities: Optional[List] = None, \
                                      overwrite: bool = False):
        lngs = self._lngs_init(lngs)
        for lng in lngs:
            self.parse_new_tree_modalities(lng, modalities, overwrite)

    def merge_attr_files(self, lngs: Union[List, str, None] = None):
        lngs = self._lngs_init(lngs)
        for lng in lngs:
            for mode in constants.MODES:
                src_dir = os.path.join(self._FLATTEN_DIR, lng, mode)
                attrs = [dir for dir in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, dir))]
                for attr in attrs:
                    src_files = sorted(glob.glob(os.path.join(src_dir, attr, '*.txt')))
                    dst_file = os.path.join(self._FLATTEN_DIR, lng, '{}.{}'.format(mode, attr))
                    cmd = 'cat {} > {}'.format(' '.join(src_files), dst_file)
                    LOGGER.info(cmd)
                    os.system(cmd)

    def close(self):
        self.pool.close()


if __name__ == '__main__':
    from multiprocessing import cpu_count

    print(os.getpid())
    # download neccesary files
    dataset = CodeSearchNet(download=True, thread_num=cpu_count())
    # flatten raw files separately
    dataset.flatten_data_all(overwrite=True)
    # # parse raw_ast into other new tree modalities
    dataset.parse_new_tree_modalities_all(overwrite=True)
    dataset.merge_attr_files()
