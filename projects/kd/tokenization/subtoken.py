import os
import re
from dataset.csn.utils.util import split_identifier
import itertools
import ujson
from ncc.utils.util_file import load_yaml
from multiprocessing import Pool, cpu_count
from dataset import LOGGER


def subtoken(in_file, out_file):
    code_flag = 'code_tokens' in in_file
    with open(in_file, 'r', encoding='utf8') as reader, open(out_file, 'w', encoding='utf8') as writer:
        for line in reader:
            line = ujson.loads(line)
            if code_flag:
                line = list(
                    itertools.chain(*[split_identifier(token, str_flag=isinstance(token, str)) for token in line]))
            line = [str.lower(token) for token in line]
            print(ujson.dumps(line), file=writer)


if __name__ == '__main__':
    yaml_file = os.path.join(os.path.dirname(__file__), 'subtoken.yml')
    args = load_yaml(yaml_file)

    in_files, out_files = [], []

    for lang, mode_files in args['dataprefs'].items():
        for mode, files in mode_files.items():
            for modality in args['modalities']:
                in_files.append(files + f'.{modality}')
                out_files.append(files + f'.{modality}' + '.subtoken')

    with Pool(cpu_count()) as mpool:
        result = [
            mpool.apply_async(subtoken, (in_file, out_file), )
            for in_file, out_file in zip(in_files, out_files)
        ]
        result = [res.get() for res in result]
