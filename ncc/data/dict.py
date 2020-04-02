# -*- coding: utf-8 -*-

import sys

# from ncc import *
from ncc.utils.constants import POS_INF
import itertools
import ujson
from typing import Any, Dict, Tuple, List, Union, Optional


class Dict(object):
    '''
    Dict for terms extraction.
    '''
    __slots__ = ('ind2label', 'label2ind', 'frequencies', 'lower', 'special')

    def __init__(self, data=None, ) -> None:
        self.ind2label = {}
        self.label2ind = {}
        self.frequencies = {}
        self.special = []  # Special entries will not be pruned.
        if data is not None:
            if type(data) == str:
                self.load_file(data)
            else:
                self.add_special_tokens(data)

    @property
    def size(self):
        return len(self.ind2label)

    def __add__(self, label: str, ind=None) -> int:
        "Add `label` in the dictionary. Use `ind` as its index if given."
        if ind is not None:
            pass
        else:
            if label in self.label2ind:
                ind = self.label2ind[label]
            else:
                ind = len(self.ind2label)

        self.ind2label[ind] = label
        self.label2ind[label] = ind

        if ind not in self.frequencies:
            self.frequencies[ind] = 1
        else:
            self.frequencies[ind] += 1

        return ind

    def _add_special_token(self, label: str, ind=None) -> None:
        "Mark this `label` and `ind` as special (i.e. will not be pruned)."
        ind = self.__add__(label, ind)
        self.special += [ind]

    def add_special_tokens(self, labels: str) -> None:
        "Mark all labels in `labels` as specials (i.e. will not be pruned)."
        for label in labels:
            self._add_special_token(label)

    def load_file(self, filename: str, ) -> None:
        '''
        Load entries from a file. such file is small
        '''
        with open(filename, "r", encoding="utf-8") as reader:
            data = ujson.loads(reader.read())
            for ind, (token, _) in enumerate(data):
                self.__add__(token, ind)

    def write_file(self, filename: str) -> None:
        "Write entries to a file."
        with open(filename, "w", encoding="utf-8") as writer:
            for label, ind in enumerate(self.label2ind):
                writer.write('{}\t{}\n'.format(label, ind))

    def lookup_ind(self, key: str, default=None) -> int:
        '''return index accroding to label'''
        return self.label2ind.get(key, default)

    def lookup_label(self, key: int, default=None) -> str:
        '''return label accroding to index'''
        return self.ind2label.get(key, default)

    def to_indices(self, labels: List[str], unk_word: str, bos_word=None, eos_word=None) -> List[int]:
        """
        Convert `labels` to indices. Use `unkWord` if not found.
        Optionally insert `bosWord` at the beginning and `eosWord` at the .
        """

        if bos_word is not None:
            indices = [self.lookup_ind(bos_word)]
        else:
            indices = []

        unk_index = self.lookup_ind(unk_word)
        indices += [self.lookup_ind(label, default=unk_index) for label in labels]

        if eos_word is not None:
            indices.append(self.lookup_ind(eos_word))

        return indices

    def to_labels(self, indices: List[int], stop_flag: int) -> List[str]:
        """
        Convert `idx` to labels.
        If index `stop` is reached, convert it and return.
        """
        labels = []

        for ind in indices:
            labels.append(self.lookup_label(ind))
            if ind == stop_flag:
                break

        return labels
