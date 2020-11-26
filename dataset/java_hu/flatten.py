from dataset.java_hu import (
    RAW_DATA_DIR, FLATTEN_DIR,
    MODES,
)
import ujson
import os


def code(src_file, dst_file):
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    with open(src_file, 'r') as reader, open(dst_file, 'w') as writer:
        for line in reader:
            print(ujson.dumps(line.rstrip('\n')), file=writer)


def code_tokens(src_file, dst_file):
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    with open(src_file, 'r') as reader, open(dst_file, 'w') as writer:
        for line in reader:
            print(ujson.dumps(line.split()), file=writer)


def docstring(src_file, dst_file):
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    with open(src_file, 'r') as reader, open(dst_file, 'w') as writer:
        for line in reader:
            print(ujson.dumps(line.rstrip('\n')), file=writer)


def docstring_tokens(src_file, dst_file):
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    with open(src_file, 'r') as reader, open(dst_file, 'w') as writer:
        for line in reader:
            docstring_tokens = line.split()
            print(ujson.dumps(docstring_tokens), file=writer)


if __name__ == '__main__':
    lang = 'java'
    _RAW_DATA_DIR, _FLATTEN_DIR = os.path.expanduser(RAW_DATA_DIR), os.path.expanduser(FLATTEN_DIR)
    for mode in MODES:
        # code
        src_file = os.path.join(_RAW_DATA_DIR, mode, 'code.original')
        dst_file = os.path.join(_FLATTEN_DIR, lang, '{}.code'.format(mode))
        code(src_file, dst_file)

        # code_tokens
        src_file = os.path.join(_RAW_DATA_DIR, mode, 'code.original_subtoken')
        dst_file = os.path.join(_FLATTEN_DIR, lang, '{}.code_tokens'.format(mode))
        code_tokens(src_file, dst_file)

        # docstring
        src_file = os.path.join(_RAW_DATA_DIR, mode, 'javadoc.original')
        dst_file = os.path.join(_FLATTEN_DIR, lang, '{}.docstring'.format(mode))
        docstring(src_file, dst_file)

        # docstring_tokens
        src_file = os.path.join(_RAW_DATA_DIR, mode, 'javadoc.original')
        dst_file = os.path.join(_FLATTEN_DIR, lang, '{}.docstring_tokens'.format(mode))
        docstring_tokens(src_file, dst_file)
