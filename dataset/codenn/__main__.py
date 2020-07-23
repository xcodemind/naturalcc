# -*- coding: utf-8 -*-

import os
import ujson
from ncc.data.constants import MODES
from ncc import LOGGER

_CODE_NN_ROOT_DIR = os.path.expanduser('~/.ncc/codenn')
RAW_DIR = os.path.join(_CODE_NN_ROOT_DIR, 'raw')
FLATTEN_DIR = os.path.join(_CODE_NN_ROOT_DIR, 'flatten')
os.makedirs(FLATTEN_DIR, exist_ok=True)


def flatten_data():
    """
    raw data:
        6602 test.txt
       52812 train.txt
        6601 valid.txt

    flatten data:
        6599 test.code
        6599 test.docstring
       52795 train.code
       52795 train.docstring
        6599 valid.code
        6599 valid.docstring
    """
    for mode in MODES:
        raw_filename = os.path.join(RAW_DIR, '{}.txt'.format(mode))
        code_filename = os.path.join(FLATTEN_DIR, '{}.code'.format(mode))
        docstring_filename = os.path.join(FLATTEN_DIR, '{}.docstring'.format(mode))
        LOGGER.info(
            'Flatten {} into code({}) and docstring({}).'.format(raw_filename, code_filename, docstring_filename))
        with open(raw_filename, 'r') as reader, \
            open(code_filename, 'w') as code_writer, open(docstring_filename, 'w') as docstring_writer:
            for idx, line in enumerate(reader):
                if line:
                    """example: [\d+]\t[\d+]\t[docstring]\t[code]\t0\n"""
                    try:
                        parsed_line = line.rstrip('\n').split('\t')
                        assert len(parsed_line) == 5, AssertionError(idx, line)
                    except AssertionError:
                        continue
                    docstring, code = parsed_line[2].strip(), parsed_line[3].strip()
                    docstring, code = docstring.replace('\\n', '\n'), code.replace('\\n', '\n')
                    print(ujson.dumps(docstring), file=docstring_writer)
                    print(ujson.dumps(code), file=code_writer)


if __name__ == '__main__':
    flatten_data()
