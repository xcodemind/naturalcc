# -*- coding: utf-8 -*-

import os

from abc import abstractmethod
from collections import namedtuple

import ncc
from ncc import LOGGER
from ncc.types import *
from ncc.utils.mp_ppool import PPool
from ncc.types import *
from ncc.utils import path

'''
This script is for downloading dataset or files.

# ====================================== CodeSearchNet ====================================== #
    Dataset: raw dataset files for Java/Javascript/PHP/GO/Ruby/Python
        Java:        https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/Java.zip
        Javascript:  https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/Javascript.zip
        PHP:         https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/PHP.zip
        GO:          https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/GO.zip
        Ruby:        https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/Java.zip
        Python:      https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/Python.zip
    Tree-Sitter: AST generation tools
        Java:        https://codeload.github.com/tree-sitter/tree-sitter-ruby/zip/master
        Javascript:  https://codeload.github.com/tree-sitter/tree-sitter-javascript/zip/master
        PHP:         https://codeload.github.com/tree-sitter/tree-sitter-php/zip/master
        GO:          https://codeload.github.com/tree-sitter/tree-sitter-go/zip/master
        Ruby:        https://codeload.github.com/tree-sitter/tree-sitter-ruby/zip/master
        Python:      https://codeload.github.com/tree-sitter/tree-sitter-python/zip/master

# ====================================== Py150 ====================================== #
    Dataset: processed dataset file for python
        Python:       http://files.srl.inf.ethz.ch/data/py150.tar.gz

# ====================================== Py150 ====================================== #
'''


class Resource(object):
    __slots__ = (
        'url',
        'size',  # raw file size (bit)
        'local'  # file name under local directory
    )

    def __init__(self, url: String_t, size: Number_t, local: String_t):
        self.url = url
        self.size = size
        self.local = path.join(local, path.basename(url))


class Dataset(object):
    """
    Args:
        _root: dataset files directory
        _thread_pool: multi-processing (Pathos) pool
        _resource: raw data files, composed of Resource namedtuple (attr: url, size)
    """

    __slots__ = (
        '_root',  # NCC root directory, download and pre-processing your data in the directory (default ~/.ncc/)
        '_cache',  # cache directory
        '_thread_pool',  # multi-processing pool build with thread_num
        '_resources'  # raw data file info

        '_RAW_DIR',  # download raw data files in this directory
        '_DATA_DIR',  # pre-processed data
    )

    def __init__(self, root: String_t = None, thread_num: Int_t = None):
        if root is None:
            root = path.join(ncc.__ROOT_DIR__, self.__class__.__name__)
        self._root = path.expanduser(root)
        path.makedirs(self._root)
        self._cache = path.expanduser(ncc.__CACHE_DIR__)
        path.makedirs(self._cache)
        self._thread_pool = PPool(thread_num)
        self._resources: Optional[Dict_t[Any_t, Resource]] = None  # register dataset online resource

    def _register_dir(self, root: String_t, *dir: Sequence_t[String_t]) -> String_t:
        """concate paths, makedir and return"""
        directory = path.safe_join(root, *dir)
        path.makedirs(directory)
        return directory

    @abstractmethod
    def _register_attrs(self):
        """write user-defined attributes of dataset"""
        self._RAW_DIR = self._register_dir(self._root, 'raw')
        self._DATA_DIR = self._register_dir(self._root, 'data')

    @abstractmethod
    def download(self):
        """download data file and others"""
        ...

    @abstractmethod
    def decompress(self):
        """decompress data file and others"""
        ...

    @abstractmethod
    def build(self):
        """default data generation method"""
        ...

    def close(self):
        self._thread_pool.close()

    def __exit__(self):
        self.close()


__all__ = (
    Resource,
    Dataset,
)
