# -*- coding: utf-8 -*-

from typing import *

from pathos.multiprocessing import (
    ProcessingPool as Pool,
    cpu_count
)


class PPool:
    """pathos multi-processing pool"""

    def __init__(self, processor_num: int = None, ):
        self.processor_num = 1 if processor_num is None \
            else min(processor_num, cpu_count())
        self._pool = Pool(self.processor_num)

    def feed(self, func: Any, params: List) -> List[Any]:
        params = tuple(zip(*params))
        result = self._pool.map(func, *params)
        return result

    def close(self):
        self._pool.close()
        self._pool.join()
        self._pool.clear()
