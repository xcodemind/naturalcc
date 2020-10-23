# -*- coding: utf-8 -*-


import os
from glob import glob

join = os.path.join
dirname = os.path.dirname
basename = os.path.basename
split = os.path.split
isdir = os.path.isdir
isfile = os.path.isfile
realpath = os.path.realpath
exists = os.path.exists
listdir = os.listdir


def expanduser(path):
    if isinstance(path) and path.startswith('~/'):
        path = os.path.expanduser(path)
    return path


def makedirs(path, exist_ok=True):
    os.makedirs(path, exist_ok=exist_ok)


def getsize(path, unit='b'):
    '''
    Args:
        path:
        unit: b - bit, B - byte, M - MB, G - GB

    Returns:
        file size
    '''
    file_sz = os.path.getsize(path)
    if unit == 'b':
        size_t = 0
    elif unit == 'B':
        size_t = 1
    elif unit == 'M':
        size_t = 2
    elif unit == 'G':
        size_t = 2
    else:
        raise NotImplementedError('unit in [b, B, M ,G]')
    file_sz = file_sz / (1024 ** size_t)
    return file_sz


# safe functions, can operate ~/ directory


def safe_join(*args):
    # if first path is '~/...', convert it to user path
    path = join(expanduser(args[0]), *args[1:])
    return path


safe_dirname = lambda path: dirname(expanduser(path))
safe_basename = basename


def safe_split(path):
    _dirname, _basename = split(path)
    _dirname = expanduser(_dirname)
    return _dirname, _basename


safe_isdir = lambda path: isdir(expanduser(path))
safe_isfile = lambda path: isfile(expanduser(path))
safe_realpath = lambda path: realpath(expanduser(path))
safe_exists = lambda path: exists(expanduser(path))
safe_makedirs = lambda path: makedirs(expanduser(path))
safe_getsize = lambda path: getsize(expanduser(path))
safe_listdir = lambda path: listdir(expanduser(path))


def safe_glob(path, suffix=None):
    if suffix is None:
        suffix = '*'
    else:
        suffix = '*.{}'.format(suffix)
    path = safe_join(path, suffix)
    return glob(path)
