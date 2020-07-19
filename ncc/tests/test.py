# -*- coding: utf-8 -*-


import os
import json
import itertools
from multiprocessing import cpu_count, Pool
from time import time


def _safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


def _find_offsets(file: str, cpu_num: int):
    with open(file, "r", encoding="utf-8") as f:
        size = os.fstat(f.fileno()).st_size
        chunk_size = size // cpu_num
        offsets = [0 for _ in range(cpu_num + 1)]
        for i in range(1, cpu_num):
            f.seek(chunk_size * i)
            _safe_readline(f)
            offsets[i] = f.tell()
        offsets[-1] = size
        return offsets


def _fast_readlines(file: str, start: int = 0, end: int = -1, func=None):
    result = []
    with open(file, 'r', encoding='UTF-8') as reader:
        reader.seek(start)
        line = reader.readline()
        while line and (reader.tell() < end):
            result.append(func(line))
            line = reader.readline()
    return result


def fast_readlines(file: str, func=None, cpu_num: int = None):
    if cpu_num is None:
        cpu_num = cpu_count()
    offsets = _find_offsets(file, cpu_num)
    with Pool(cpu_num) as thread_pool:
        jobs = [
            thread_pool.apply_async(_fast_readlines, (file, offsets[worker_id], offsets[worker_id + 1], func))
            for worker_id in range(cpu_num)
        ]
        multiple_results = [job.get() for job in jobs]
    return multiple_results


if __name__ == '__main__':
    file = '~/.ncc/py150/raw/python100k_train.json'
    file = os.path.expanduser(file)
    start = time()
    out = fast_readlines(file, func=json.loads, cpu_num=100)
    out = list(itertools.chain(*out))
    print(time() - start)
    # print(len(out))
    """
    20.605948209762573
    103254
    """

    start = time()
    out = []
    with open(file, 'r') as reader:
        # out = [json.loads(line) for line in reader.readlines()]
        line = reader.readline()
        while line:
            out.append(json.loads(line))
            line = reader.readline()
    print(time() - start)
    # print(len(out))
    """
    15.215238571166992
    103266
    """
