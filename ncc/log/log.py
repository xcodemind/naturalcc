# -*- coding: utf-8 -*-

import logging


def get_logger(level=logging.DEBUG) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format='[%(asctime)-15s] %(levelname)7s >> %(message)s (%(filename)s:%(lineno)d, %(funcName)s())',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger(__name__)
    return logger
