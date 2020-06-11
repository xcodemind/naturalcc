# -*- coding: utf-8 -*-

import os

import ncc
from ncc import LOGGER
from ncc.types import *
from ncc.utils.mp_ppool import PPool

from . import *

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


class Dataset(object):
    __slots__ = ('_root', '_cache', '_thread_pool')

    def __init__(self, root: String_t = None, thread_num: Int_t = None):
        if root is None:
            root = path_join_func(ncc.__ROOT_DIR__, self.__class__.__name__)
        self._root = expanduser_func(root)
        makedirs_func(self._root)
        self._cache = expanduser_func(ncc.__CACHE_DIR__)
        makedirs_func(self._cache)
        self._thread_pool = PPool(thread_num)

    def close(self):
        self._thread_pool.close()

    def __exit__(self):
        self.close()

    # @staticmethod
    # def codesearchnet(self):
    #     resources = [
    #         ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    #         ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    #         ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    #         ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    #     ]
    #     raw_file_sizes = {
    #         'go': 487525935,
    #         'java': 1060569153,
    #         'javascript': 1664713350,
    #         'php': 851894048,
    #         'python': 940909997,
    #         'ruby': 111758028,
    #     }
