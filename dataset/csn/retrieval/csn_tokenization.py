import os
import itertools
import ujson
from dataset.csn import (
    MODES, FLATTEN_DIR
)
from dataset import LOGGER
import sentencepiece as spm
from dataset.csn.utils.util import split_identifier


# def split_code_tokens(src_file, tgt_file):
#     with open(src_file, 'r') as reader, open(tgt_file, 'w') as writer:
#         for line in reader:
#             code_tokens = ujson.loads(line)
#             code_tokens = [split_identifier(token, str_flag=False) for token in code_tokens]
#             code_tokens = itertools.chain(*code_tokens)
#             code_tokens = [token for token in code_tokens if token is not None and len(token) > 0]
#             print(ujson.dumps(code_tokens), file=writer)

def joint_code_tokens(src_file, tgt_file):
    with open(src_file, 'r') as reader, open(tgt_file, 'w') as writer:
        for line in reader:
            code_tokens = ujson.loads(line)
            code_tokens = [split_identifier(token, str_flag=False) for token in code_tokens]
            code_tokens = itertools.chain(*code_tokens)
            code_tokens = [token for token in code_tokens if token is not None and len(token) > 0]
            code_tokens = ' '.join(code_tokens).lower()
            print(ujson.dumps(code_tokens), file=writer)


def joint_docstring_tokens(src_file, tgt_file):
    with open(src_file, 'r') as reader, open(tgt_file, 'w') as writer:
        for line in reader:
            docstring_tokens = ujson.loads(line)
            docstring_tokens = ' '.join(docstring_tokens).lower()
            print(ujson.dumps(docstring_tokens), file=writer)


if __name__ == '__main__':
    lang = 'javascript'
    for mode in MODES:
        file_prefix = os.path.join(FLATTEN_DIR, lang, mode)

        # split code_tokens
        code_tokens_files = file_prefix + '.code_tokens'
        splited_code_tokens_files = file_prefix + '.joint.code_tokens'
        joint_code_tokens(code_tokens_files, splited_code_tokens_files)

        # BPE docstring_tokens
        docstring_tokens_files = file_prefix + '.docstring_tokens'
        joint_docstring_tokens_files = file_prefix + '.joint.docstring_tokens'
        joint_docstring_tokens(docstring_tokens_files, joint_docstring_tokens_files)
