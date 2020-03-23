# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from src import *


class Trainer(object):

    def __init__(self, config: Dict, ) -> None:
        LOGGER.debug('building {}...'.format(self))
        self.config = config

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return str(self)

    @abc.abstractmethod
    def train(self, *args: Any, **kwargs: Any, ) -> Any:
        LOGGER.info('{} train...'.format(self))

    @abc.abstractmethod
    def train_al(self, *args: Any, **kwargs: Any, ) -> Any:
        LOGGER.info('{} adversarial learning train...'.format(self))

    @abc.abstractmethod
    def meta_train(self, *args: Any, **kwargs: Any, ) -> Any:
        LOGGER.info('{} meta_train...'.format(self))
