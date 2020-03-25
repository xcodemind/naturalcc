# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

from typing import *

import glob
from ncc.utils.constants import MODES
import itertools
import ujson


def load_file(filename):
    with open(filename, 'r') as reader:
        return [ujson.loads(line.strip()) for line in reader.readlines() if len(line.strip()) > 0]


def main():
    ast_dir = '/data/wanyao/yang/ghproj_d/GitHub/datasetv2/key/100_small/python/ast'
    files = {mode: [fl for fl in glob.glob('{}/*'.format(ast_dir, )) if mode in fl] for mode in MODES}
    data = {
        mode: sorted(list(itertools.chain(*[load_file(fl) for fl in files[mode]])), key=lambda ast_tree: len(ast_tree))
        for mode in MODES
    }


if __name__ == '__main__':
    main()
