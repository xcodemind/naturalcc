# -*- coding: utf-8 -*-

import os
import ujson
from glob import glob
import itertools
from dataset.codesearchnet.utils import util
from dataset.codesearchnet.utils.constants import LANGUAGES, MODES

import gzip

for lang, mode in itertools.product(LANGUAGES, MODES):
    outfile = os.path.expanduser('~/{}_{}.non_ascii'.format(lang, mode))
    print(outfile)
    with open(outfile, 'w') as writer:
        raw_files = '~/.ncc/CodeSearchNet/raw_unzip/{}/{}_{}_*.jsonl.gz'.format(lang, lang, mode)
        raw_files = os.path.expanduser(raw_files)
        raw_files = glob(raw_files)
        for file in raw_files:
            reader = gzip.GzipFile(file, 'r')
            for line in reader:
                code_snippet = ujson.loads(line)
                code, docstring, url = code_snippet['code'], code_snippet['docstring'], code_snippet['url']
                if not str.isascii(code) or not str.isascii(docstring):
                    # writer.write(ujson.dumps({'code': code, 'docstring': docstring, 'url': url}) + '\n')
                    print(ujson.dumps({'code': code, 'docstring': docstring, 'url': url}, ensure_ascii=False),
                          file=writer)
