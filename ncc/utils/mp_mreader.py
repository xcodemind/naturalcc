# -*- coding: utf-8 -*-

from typing import *

import os
from pathos.multiprocessing import cpu_count
from multiprocessing.pool import Pool
from .mp_ppool import PPool


def _safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


def _read(file: str, func: Any, param: Optional[List] = None, start: int = 0, end: int = -1) -> List:
    result = []
    with open(file, 'r', encoding='UTF-8') as reader:
        reader.seek(start)
        line = reader.readline()
        while line and (reader.tell() < end):
            if param:
                result.append(func(line, *param))
            else:
                result.append(func(line))  # param == None
            line = reader.readline()
    return result


def _find_offsets(file: str, cpu_num: int) -> List:
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


def readline(file: str, func: Any, params: Optional[List[Sequence]] = None, cpu_num: int = None) -> List:
    if cpu_num is None:
        cpu_num = cpu_count()
    if params is None:
        params = [None] * cpu_num
    pool = PPool(cpu_num)
    offsets = _find_offsets(file, cpu_num)
    _read_params = [
        (file, func, params[idx], offsets[idx], offsets[idx + 1],)
        for idx in range(cpu_num)
    ]
    result = pool.feed(_read, _read_params)
    return result


def _fast_readlines(file: str, start: int = 0, end: int = -1, func=None) -> List:
    result = []
    with open(file, 'r', encoding='UTF-8') as reader:
        reader.seek(start)
        line = reader.readline()
        while line and (reader.tell() < end):
            result.append(func(line))
            line = reader.readline()
    return result


def fast_readlines(file: str, func=None, cpu_num: int = None) -> List:
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