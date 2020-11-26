# -*- coding: utf-8 -*-

import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)-15s] %(levelname)7s >> %(message)s (%(filename)s:%(lineno)d, %(funcName)s())',
    datefmt='%Y-%m-%d %H:%M:%S',
)
LOGGER = logging.getLogger(__name__)

import sys

# backwards compatibility to support `from fairseq.meters import AverageMeter`
from ncc.logging import meters, metrics, progress_bar  # noqa

sys.modules['ncc.meters'] = meters
sys.modules['ncc.metrics'] = metrics
sys.modules['ncc.progress_bar'] = progress_bar

# import ncc.criterions  # noqa
# import ncc.model  # noqa
# import ncc.modules  # noqa
# import ncc.optim  # noqa
# import ncc.optim.lr_scheduler  # noqa
# import ncc.utils.pdb as pdb # noqa
# import ncc.tasks  # noqa

# import ncc.benchmark  # noqa

# ============== NCC Library info ============== #
# library properties
__NAME__ = 'NCC'
__FULL_NAME__ = 'Natural Code and Comment Lib.'
__DESCRIPTION__ = '''
NCC (Natural Code and Comment Lib.) is developed for Programming Language Processing and Code Analysis.
'''
# __VERSION__ = ‘x,y(a,b,c)’, e.g. '1.2c'
# x for framework update or a considerable update,
# y for internal update, e.g. naturalcodev3 is our 3rd Version of coding,
__VERSION__ = '0.3'
__AUTHORS__ = [
    'YangHE', 'YaoWAN',  # main authors, sharing same contributions
    # other authors
    'JingdongSUN', 'WeiHUA',
]
__AFFILIATIONS__ = [
    'Univ. of Tech. Sydney (UTS)', 'Z.J. Univ. (ZJU)',  # main affiliations, sharing same contributions
    # other affiliations
    'Wuhan Univ. of Tech. (WHUT)',
]
__BIRTH__ = '2018/06-Now'

import os

__ROOT_DIR__ = '~/.ncc'
__CACHE_DIR__ = '~/.ncc/cache'
# ============== NCC Library info ============== #
