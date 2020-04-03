# -*- coding: utf-8 -*-
import sys

# from ncc import *
from ncc import LOGGER
from typing import Any, Dict, Tuple, List, Union, Optional
from ncc.data.dict import Dict as _Dict


class TokenDicts(object):
    __slots__ = ('dicts',)

    def __init__(self, dict_filenames: Dict, ) -> None:
        self.dicts = {}
        for key, dict_fl in dict_filenames.items():
            LOGGER.info("in TokenDicts ,key:{} load :{}".format(key,dict_fl))
            self.dicts[key] = self._load_dict(dict_fl)

    def _load_dict(self, dict_file=None) -> Union[_Dict, None]:
        if dict_file is None:
            return None
        else:
            token_dict = _Dict()
            token_dict.load_file(dict_file)
            return token_dict

    @property
    def size(self) -> Dict:
        return {key: token_dict.size for key, token_dict in self.dicts.items()}

    def __str__(self) -> str:
        return 'TokenDicts({})'.format(''.join(['{}_dict: {}, '.format(key, size) for key, size in self.size.items()]))

    def __repr__(self):
        return str(self)

    def __getitem__(self, key: str, ) -> _Dict:
        return self.dicts[key]
