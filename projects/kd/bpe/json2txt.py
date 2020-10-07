import os
import re
import ujson
import itertools
from dataset.csn.utils.util import split_identifier
from ncc.utils.util_file import load_yaml
from multiprocessing import Pool, cpu_count
from dataset import LOGGER

_LF_regex = re.compile(r"\\n")
_CR_regex = re.compile(r"\\r")
_whitespace_regex = re.compile(r"[ \t\n]+")


def normalize_program(fn: str):
    if not isinstance(fn, (str, bytes)):
        LOGGER.error(f"normalize_program got non-str: {type(fn)}, {fn}")
    fn = _LF_regex.sub(r"[LF]", fn)
    fn = _CR_regex.sub(r"[CR]", fn)
    fn = _whitespace_regex.sub(" ", fn)
    return fn


def jsonstring2txt(in_file, out_file):
    concate_end = 'java_hu' in in_file or 'python_wan' in in_file
    with open(in_file, 'r', encoding='utf8') as reader, open(out_file, 'w', encoding='utf8') as writer:
        for line in reader:
            line = ujson.loads(line).strip()
            line = normalize_program(line)
            if concate_end and line[-1] == '.' and line[-2] != ' ':
                line = line[:-2] + '.'
            print(line, file=writer)


def jsonlist2txt(in_file, out_file):
    concate_end = 'java_hu' in in_file or 'python_wan' in in_file
    code_flag = 'code_tokens' in in_file
    with open(in_file, 'r', encoding='utf8') as reader, open(out_file, 'w', encoding='utf8') as writer:
        for line in reader:
            line = ujson.loads(line)
            if code_flag:
                line = list(
                    itertools.chain(*[split_identifier(token, str_flag=isinstance(token, str)) for token in line]))

            line = ' '.join(line).lower()
            line = normalize_program(line)
            if concate_end and line[-1] == '.' and line[-2] != ' ':
                line = line[:-2] + '.'
            print(line, file=writer)


if __name__ == '__main__':
    yaml_file = os.path.join(os.path.dirname(__file__), 'json2txt.yml')
    args = load_yaml(yaml_file)
    special_symbols = ['[EOL]']

    in_files, out_files = [], []

    for lang, mode_files in args['dataprefs'].items():
        for mode, files in mode_files.items():
            for modality in args['modalities']:
                in_files.append(files + f'.{modality}')
                out_files.append(files + f'.{modality}' + '.txt')

    with Pool(cpu_count()) as mpool:
        # result = [
        #     mpool.apply_async(jsonstring2txt, (in_file, out_file), )
        #     for in_file, out_file in zip(in_files, out_files)
        # ]
        result = [
            mpool.apply_async(jsonlist2txt, (in_file, out_file), )
            for in_file, out_file in zip(in_files, out_files)
        ]
        result = [res.get() for res in result]
