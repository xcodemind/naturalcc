# -*- coding: utf-8 -*-

from ncc import *
import time
from uuid import uuid1
import hashlib


def time_id(debug) -> str:
    # return str(time.strftime('%Y-%b-%d', time.localtime(time.time())))
    if debug:
        return str(time.strftime('%Y-%b-%d', time.localtime(time.time())))
    else:
        return str(time.strftime('%Y-%b-%d-%H-%M-%S', time.localtime(time.time())))


def mac_id() -> str:
    return str(uuid1())


def md5_id() -> str:
    md5 = hashlib.md5()
    md5.update(str(time.strftime('%Y-%b-%d_%H-%M-%S', time.localtime(time.time()))).encode('utf-8'))
    return str(md5.hexdigest())


if __name__ == '__main__':
    # print(time_id())
    # print(mac_id())
    print(md5_id())
    print(md5_id())
