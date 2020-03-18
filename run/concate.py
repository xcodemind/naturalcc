# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

from typing import *
import os
import glob

# java_javascript_php_python_ruby.ast.dict
# java_javascript_php_python_ruby.border.dict
# java_javascript_php_python_ruby.center.dict
# java_javascript_php_python_ruby.code_tokens.dict
# java_javascript_php_python_ruby.comment.dict
# java_javascript_php_python_ruby.docstring_tokens.dict
# java_javascript_php_python_ruby.method.dict
# java_javascript_php_python_ruby.sbt2.dict
# java_javascript_php_python_ruby.sbtao.dict
# java_javascript_php_python_ruby.sbt.dict
# java_javascript_php_python_ruby.tok.dict

# java8javascript8php8python_ruby

def main():
    MODES = ['train', 'valid', 'test']
    # DATASET_DIR = '/data/wanyao/yang/ghproj_d/GitHub/datasetv2/key/100_small'
    DATASET_DIR =  '/data/wanyao/ghproj_d/naturalcodev2/datasetv2/key/100_small'
    # languages = sorted(['python', 'php', 'java', 'javascript'])
    languages = sorted(['python', 'php', 'java', 'javascript','go'])
    target_dir = os.path.join(DATASET_DIR, '8'.join(languages))
    os.makedirs(target_dir, exist_ok=True)
    KEYS = ['tok', 'docstring_tokens', 'path', 'index', 'code_tokens', 'docstring', 'comment', 'code', ]

    for key in KEYS:
        trg_key_dir = os.path.join(target_dir, key)
        os.makedirs(trg_key_dir, exist_ok=True)

        for mode in MODES:
            start_ind = 0
            while True:
                src_files = [os.path.join(DATASET_DIR, lng, key, '{}_{}_{}.txt'.format(lng, mode, start_ind)) \
                             for lng in languages]
                src_files = [fl for fl in src_files if os.path.exists(fl)]
                if len(src_files) == 0:
                    break
                else:
                    cmd = 'cat {} > {}'.format(
                        ' '.join(src_files),
                        os.path.join(target_dir, key, '{}_{}_{}.txt'.format('8'.join(languages), mode, start_ind))
                    )
                    print(cmd)
                    os.system(cmd)
                    start_ind += 1


if __name__ == '__main__':
    main()
