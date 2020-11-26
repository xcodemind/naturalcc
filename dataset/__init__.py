# -*- coding: utf-8 -*-

import os
import re
import socket
import getpass
from ncc import LOGGER

HOSTNAME = socket.gethostname()
USERNAME = getpass.getuser()
# register your hostname or username
if HOSTNAME in ['GS65', 'node14'] or re.match(r'pytorch-.*', HOSTNAME) is not None \
    or USERNAME in ['hust_xhshi_1']:
    DEFAULT_DIR = '~/.ncc'
else:
    DEFAULT_DIR = '/export/share/jianguo/scodebert/'

DEFAULT_DIR = os.path.expanduser(DEFAULT_DIR)
LIBS_DIR = os.path.join(os.path.dirname(__file__), 'tree-sitter-libs')
LOGGER.debug('Host Name: {}; User Name: {}; Default data directory: {}'.format(HOSTNAME, USERNAME, DEFAULT_DIR))

__all__ = (
    HOSTNAME, USERNAME,
    DEFAULT_DIR, LIBS_DIR,
    LOGGER,
)
