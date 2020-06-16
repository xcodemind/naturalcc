# -*- coding: utf-8 -*-

import ujson
import wget

from dataset.utils.dataset import Resource, Dataset

from ncc import LOGGER
from ncc.types import *
from ncc.utils import path


class Py150(Dataset):
    """
    # ====================================== Py150 ====================================== #
    Dataset: processed Py150 data files of Python, ast with Deep Search First traversal
        # language   # URL                                                                       # size
        Python       https://files.sri.inf.ethz.ch/data/py150.tar.gz                             526642289
    """
    __slots__ = (
        '_resources',

        '_RAW_DIR',
        '_RAW_FILES',
        '_DATA_DIR',
        '_DATA_FILES'
    )

    def _register_attrs(self):
        super()._register_attrs()

        self._resources = Resource(
            url='https://files.sri.inf.ethz.ch/data/py150.tar.gz',
            size=526642289,
            local=self._RAW_DIR,
        )
        self._RAW_FILES = path.safe_glob(self._RAW_DIR, suffix='json')
        self._DATA_FILES = [
            path.join(self._DATA_DIR, path.basename(filename))
            for filename in self._RAW_FILES
        ]

    def __init__(self, root: String_t = None, thread_num: Int_t = None,
                 overwrite: Bool_t = True):
        super().__init__(root, thread_num)
        self._register_attrs()

    def download(self, overwrite: Bool_t = False):
        # if 1) file doesnt exist, or 2) overwrite download file, or 3) the size of raw file doesnt match
        if (not path.exists(self._resources.local)) or overwrite or \
                (not path.getsize(self._resources.local) == self._resources.size):
            url, local = self._resources.url, self._resources.local
            LOGGER.info('Download py150.tar.gz from {}'.format(url))
            wget.download(url=url, out=local)
        # raw file size check
        assert path.getsize(self._resources.local) == self._resources.size, \
            RuntimeError('the size of raw file doesnt match, pls download again')

    def decompress(self, overwrite: Bool_t = False):
        assert path.exists(self._resources.local)
        import tarfile
        with tarfile.open(self._resources.local) as reader:
            for out_filename in reader.getnames():
                # extract json files
                if str.endswith(out_filename, '.json') and (
                        not path.exists(path.join(self._RAW_DIR, out_filename)) or overwrite
                ):
                    reader.extract(out_filename, path=self._RAW_DIR)

    def convert(self, raw_file: String_t, data_file: String_t):
        """extract leaf node's value into a new node"""

        def _convert(ast: Sequence_t[Dict_t]) -> Sequence_t[Dict_t]:
            increase_by = {}  # count of how many idx to increase the new idx by:
            # each time there is a value node
            cur = 0
            for i, node in enumerate(ast):
                increase_by[i] = cur
                if "value" in node:
                    cur += 1

            new_ast = []
            for i, node in enumerate(ast):
                inc = increase_by[i]
                if "value" in node:
                    child = [i + inc + 1]
                    if "children" in node:
                        child += [n + increase_by[n] for n in node["children"]]
                    new_ast.append({"type": node["type"], "children": child})
                    new_ast.append({"value": node["value"]})
                else:
                    if "children" in node:
                        node["children"] = [n + increase_by[n] for n in node["children"]]
                    new_ast.append(node)
            return new_ast

        with open(raw_file, 'r') as reader, open(data_file, 'w') as writer:
            for line in reader:
                raw_data = ujson.loads(line)
                data = _convert(raw_data)
                writer.write(ujson.dumps(data) + '\n')

    def build(self, overwrite: Bool_t = False):
        # self.download()
        # self.decompress()
        for raw_file, data_file in zip(self._RAW_FILES, self._DATA_FILES):
            self.convert(raw_file, data_file)


if __name__ == '__main__':
    dataset = Py150()
    dataset.build()
