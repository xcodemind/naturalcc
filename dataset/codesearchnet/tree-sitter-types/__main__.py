# -*- coding: utf-8 -*-

import os
import wget
import ujson

from ncc.types import *
from ncc.utils import path
from dataset.codesearchnet.utils.constants import LANGUAGES

NODE_TYPES_FILES = {
    lang: {
        'url': 'https://raw.githubusercontent.com/tree-sitter/tree-sitter-{}/master/src/node-types.json'.format(lang),
        'dir': os.path.join(os.path.dirname(__file__), '{}-node-types.json'.format(lang)),
    }
    for lang in LANGUAGES
}


def download():
    for lang in LANGUAGES:
        if not os.path.exists(NODE_TYPES_FILES[lang]['dir']):
            wget.download(NODE_TYPES_FILES[lang]['url'], out=NODE_TYPES_FILES[lang]['dir'])


OPERATOR = set([])


class Types:
    def __init__(self, file):
        self.family_types = []
        with open(file, 'r') as reader:
            self.types = ujson.load(reader)
        self.run()

    def run(self):
        for type in self.types:
            if (type['named'] == False) and ('type' in type) and (not str.isalnum(type['type'])):
                OPERATOR.add(type['type'])


if __name__ == '__main__':
    # download()

    for lang in LANGUAGES:
        types = Types(NODE_TYPES_FILES[lang]['dir'])
