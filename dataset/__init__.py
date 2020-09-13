# -*- coding: utf-8 -*-

import os
import socket
import getpass
from ncc import LOGGER

HOSTNAME = socket.gethostname()
USERNAME = getpass.getuser()
# register your hostname or username
if HOSTNAME in ['GS65'] or USERNAME in ['hust_xhshi_1']:
    DEFAULT_DIR = '~/.ncc'
else:
    DEFAULT_DIR = '/export/share/jianguo/scodebert/'

DEFAULT_DIR = os.path.expanduser(DEFAULT_DIR)
LOGGER.info('Host Name: {}; User Name: {}; Default data directory: {}'.format(HOSTNAME, USERNAME, DEFAULT_DIR))

__all__ = (
    HOSTNAME, USERNAME,
    DEFAULT_DIR, LOGGER,
)
