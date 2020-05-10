# -*- coding: utf-8 -*-

import os
import ujson

S_SEP = '<S_SEP>'
S_CLS = '<CLS>'
S_MASK = '<MASK>'
output_prefixs = ['train', 'valid', 'test']  # insert spec tokens for all datasets.
PREPROCESS = 'preprocess'
prefix = 'pref'


def file_name(prefix, lang):
    fname = prefix
    if lang is not None:
        fname += ".{lang}".format(lang=lang)
    return fname


def dest_path(args, prefix, lang):
    return os.path.join(args['preprocess']['destdir'], file_name(prefix, lang))


def insert_sep_token(args, input_prefix, output_prefix, lang):
    output_text_file = dest_path(args,
                                 output_prefix,
                                 # + ".{}-{}".format(args['preprocess']['source_lang'], args['preprocess']['target_lang'])
                                 lang,
                                 )
    if lang == 'code':
        #     insert <S_SEP> to .code files
        with open(output_text_file, 'w') as output_file:
            with open(file_name(input_prefix, lang), 'r') as input_file:
                for line in input_file.readlines():
                    ln = ujson.loads(line)
                    for count in range(10, 1, -1):  # to handle duplicate '\n'
                        ln = ln.replace('\n', S_SEP, count)
                        ln = S_CLS + ln  # profix <CLS>
                    output_file.write(ujson.dumps(ln) + '\n')
    else:
        raise ValueError(
            "dataset 'code' supported only 2020.0506"
        )


def insert_sep_tokens(args, lang='code'):
    for idx in output_prefixs:
        insert_sep_token(args, args[PREPROCESS][idx + prefix], idx, lang)
