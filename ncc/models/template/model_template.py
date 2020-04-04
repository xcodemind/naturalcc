# -*- coding: utf-8 -*-
from torch.nn import Module
from typing import Dict, Any
import abc


class IModel(Module):
    '''
    interface for model
    '''

    @abc.abstractmethod
    def __str__(self) -> Any:
        pass

    def __repr__(self) -> str:
        return str(self)

    @abc.abstractmethod
    def train_pipeline(self, batch_data: Dict, ) -> Any:
        pass

    @abc.abstractmethod
    def eval_pipeline(self, batch_data: Dict, ) -> Any:
        pass

    @abc.abstractmethod
    def train_sl(self, *args: Any, **kwargs: Any, ) -> Any:
        pass

    @abc.abstractmethod
    def train_ft(self, *args: Any, **kwargs: Any, ) -> Any:
        pass

    @abc.abstractmethod
    def meta_train(self, *args: Any, **kwargs: Any, ) -> Any:
        pass

    @abc.abstractmethod
    def meta_val(self, *args: Any, **kwargs: Any, ) -> Any:
        pass
