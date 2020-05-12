# -*- coding: utf-8 -*-

from typing import *

import time
from ncc.multiprocessing.ppool import PPool
from ncc.multiprocessing.mpool import MPool


class Foo:
    def func(self, sleep_time: int, idx: int):
        print(sleep_time)
        time.sleep(sleep_time)
        return idx


if __name__ == '__main__':
    var = Foo()
    ppool = PPool(2)
    mpool = MPool(2)
    args = list(zip([2] * 20, range(20)))
    print(args)

    start = time.time()
    ppool.feed(var.func, args)
    print(time.time() - start)
    exit()

    start = time.time()
    mpool.feed(Foo.func, args)
    print(time.time() - start)
