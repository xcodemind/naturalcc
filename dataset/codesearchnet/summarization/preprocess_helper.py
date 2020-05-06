import os
import ujson
import os



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
                    line_ = ujson.loads(line)
                    for count in [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:  # to handle duplicate '\n'
                        line_ = line_.replace('\n', ' <S_SEP> ', count)
                    output_file.write(ujson.dumps(line_) + '\n')
    else:
        raise ValueError(
            "code dataset supported only 2020.0506"
        )

def insert_sep_tokens(args,lang='code'):
    preprocess='preprocess'
    output_prefix=['train','valid','test']
    prefix='pref'
    for idx in output_prefix:
        insert_sep_token(args,args[preprocess][idx+prefix],idx,lang)

