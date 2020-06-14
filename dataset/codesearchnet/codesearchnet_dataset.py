# -*- coding: utf-8 -*-

import wget
import ujson
from dataset.utils.dataset import Resource, Dataset

from ncc import LOGGER
from ncc.types import *
from ncc.utils import path
from ncc.utils.constants import LANGUAGES


def lngs_init(lngs: Union[Sequence_t, String_t, None]) -> Sequence_t:
    if lngs is None:
        return LANGUAGES
    elif isinstance(lngs, String_t):
        return [lngs]
    elif isinstance(lngs, Sequence_t):
        for lng in lngs:
            assert lng in LANGUAGES, RuntimeError('{} is not in {}'.format(lng, LANGUAGES))
    else:
        raise NotImplementedError('Only None/List/str is available.')
    return lngs


def load_json(filename: String_t = './data.json') -> Tuple[Dict_t[String_t, Resource]]:
    def json2dict(data: Dict_t[String_t, Dict_t]) -> Dict_t[String_t, Resource]:
        data = {
            key: Resource(url=value['url'], size=value['size'] if 'size' in value else -1)
            for key, value in data.items()
        }
        return data

    current_dir = path.dirname(__file__)
    filename = path.join(current_dir, filename)
    assert path.exists(filename)
    with open(filename, 'r') as reader:
        context = reader.read().strip()
        data = ujson.loads(context)
    data_info, libs_info = data['raw'], data['libs']
    data_info: Dict_t[String_t, Resource] = json2dict(data_info)
    libs_info: Dict_t[String_t, Resource] = json2dict(libs_info)
    return data_info, libs_info


class CodeSearchNet(Dataset):
    """
    # ====================================== CodeSearchNet ====================================== #
    Dataset: raw CodeSearchNet data files of Java/Javascript/PHP/GO/Ruby/Python
       # language   # URL                                                                       # size
       java:        https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip			1060569153
       javascript:  https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/javascript.zip	1664713350
       php:         https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/php.zip			851894048
       go:          https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/go.zip			487525935
       ruby:        https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/ruby.zip			111758028
       python:      https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip		940909997
    Tree-Sitter: AST generation tools, TreeSitter repositories from Github can be updated, therefore their size is capricious
       # language   # URL
       Java:        https://codeload.github.com/tree-sitter/tree-sitter-java/zip/master
       Javascript:  https://codeload.github.com/tree-sitter/tree-sitter-javascript/zip/master
       PHP:         https://codeload.github.com/tree-sitter/tree-sitter-php/zip/master
       GO:          https://codeload.github.com/tree-sitter/tree-sitter-go/zip/master
       Ruby:        https://codeload.github.com/tree-sitter/tree-sitter-ruby/zip/master
       Python:      https://codeload.github.com/tree-sitter/tree-sitter-python/zip/master
    """

    __slots__ = (
        '_attrs', '_resources', '_ast_modalities', '_tree_sitter_libs',
        '_RAW_DIR', '_LIB_DIR',
    )

    def _register_attrs(self):
        self._attrs: Set_t = {'code', 'code_tokens', 'docstring', 'docstring_tokens', 'func_name', 'original_string',
                              'path', 'repo', 'sha', 'url', 'partition', 'language'}
        self._ast_modalities = {'path', 'sbt', 'sbtao', 'bin_ast'}
        """
        download original data from
        https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{python,java,go,php,ruby,javascript}.zip
        """
        self._resources, self._tree_sitter_libs = load_json()

        self._RAW_DIR = path.join(self._root, 'raw')  # raw data directory
        self._LIB_DIR = path.join(self._root, 'libs')  # TreeSitter libraries directory

    def __init__(self, root: String_t = None, thread_num: Int_t = None):
        super().__init__(root, thread_num)

    def download(self, lng: String_t, overwrite: Bool_t = True):
        def _download_lib():
            path.safe_makedirs(self._LIB_DIR)
            lib_file = path.join(self._LIB_DIR, '{}.zip'.format(lng))
            if path.safe_exists(lib_file) and overwrite:
                pass
            else:
                url, out = self._tree_sitter_libs[lng].url, self._tree_sitter_libs[lng].out
                LOGGER.info('Download Tree-Sitter Library {} from {}'.format(lib_file, url))
                wget.download(url=url, out=out)


if __name__ == '__main__':
    dataset = CodeSearchNet()
