# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.path.abspath('.'))

import time
import torch

import json
from typing import *
import itertools

import multiprocessing as mp

from src.data.dict import Dict as _Dict
from src.utils.constants import *


def save_json(output, output_path):
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(output, json_file, ensure_ascii=False)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as json_file:
        aa = json_file.readlines()[0]
        output = json.loads(aa)
    return output


def now() -> str:
    return time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))


def mkdir(dir_path):
    # create dir with recursion
    if os.path.exists(dir_path):
        pass
    else:
        try:
            os.makedirs(dir_path)
        except Exception as err:
            print(str(err).strip())


def merge_dict(dicts: List[Dict]) -> Dict:
    keys = list(set(list(itertools.chain(*[list(dct.keys()) for dct in dicts]))))
    new_dict = dict()
    for key in keys:
        new_dict[key] = []
        for dct in dicts:
            if key in dct:
                new_dict[key].append(dct[key])
            else:
                continue
    return new_dict


# ================ write/read dict elements ================ #
def load_flatten_dict(file_dir: str, load_keys=List[str],
                      mpool=None) -> Dict:
    '''

    :param file_dir:
    :param load_keys:
    :param mpool:
    :return:
    '''
    filenames = {
        key: os.path.join(file_dir, '{}.pt'.format(key, ))
        for key in load_keys
    }
    if mpool is None:
        return {
            key: torch.load(fl_name)
            for key, fl_name in filenames.items()
        }

    else:
        key_data = mpool.map(torch.load, filenames.values())
        return {
            key: data
            for key, data in zip(filenames.keys(), key_data)
        }


def dump_flatten_dict(obj: Dict, file_dir: str,
                      mpool=None) -> None:
    '''
    flatten dict and then write them to pt files with multi-processing
    :param obj:
    :param file_dir:
    :param mpool: multi-processing flag
    :return:
    '''
    mkdir(file_dir)
    params = [{
        'obj': value,
        'filename': os.path.join(file_dir, '{}.pt'.format(key, )),
    } for key, value in obj.items()]
    if mpool is None:
        for prms in params:
            torch.save(*prms)
    else:
        mpool.map(mp_torch_save, params)


# ================ write/read dict elements ================ #

def mp_torch_save(params: Dict) -> None:
    # for multi-processing
    obj, filename, = params.values()
    torch.save(obj, filename)


def flatten_dict_list(dict_list: List[Dict]) -> Dict:
    '''
    dict list -> dict
    :param dict_list:
    :return:
    '''
    dict_keys = dict_list[0].keys()
    dict_values = zip(*map(lambda dct: dct.values(), dict_list))
    new_dict = {
        key: list(value)
        for key, value in zip(dict_keys, dict_values)
    }
    return new_dict


def mpool(core_num=None) -> mp.Pool:
    '''
    create multi-processing pool
    :param core_num:
    :return:
    '''
    # maximize processor number
    if core_num is None:
        core_num = mp.cpu_count()
    else:
        core_num = core_num if mp.cpu_count() > core_num else mp.cpu_count()
    mp_pool = mp.Pool(processes=core_num)
    return mp_pool


def extend_dict(words: List, dict: Any):
    ids = []
    oovs = []
    for w in words:
        # print('w: ', w)
        idx = dict.lookup_ind(w, default=UNK)
        # print('idx: ', idx)
        if idx == UNK:  # If w is OOV
            if w not in oovs:  # Add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
            # print('dict.size:', dict.size, ' oovs: ', oovs)
            ids.append(dict.size + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(idx)
    # print('ids: ', ids)
    # print('oovs: ', oovs)
    return ids, oovs


def extend_dict_with_oovs(words, dict, oovs):
    ids = []
    for w in words:
        idx = dict.lookup_ind(w, default=UNK)
        if idx == UNK:  # If w is an OOV word
            if w in oovs:  # If w is an in-article OOV
                dict_idx = dict.size + oovs.index(w)  # Map to its temporary article OOV number
                ids.append(dict_idx)
            else:  # If w is an out-of-article OOV
                ids.append(UNK)  # Map to the UNK token id
        else:
            ids.append(idx)
    return ids


def clean_up_sentence(sentence: torch.Tensor, remove_UNK=False, remove_EOS=False):
    # cut PAD at the sentence's end
    sentence = sentence.tolist()
    if PAD in sentence:
        sentence = sentence[:sentence.index(PAD)]
    # cut EOS at the sentence's end
    if EOS in sentence:
        sentence = sentence[:sentence.index(EOS) + 1]
    # filter UNK
    if remove_UNK:
        sentence = filter(lambda x: x != UNK, sentence)
    # remove EOS
    if remove_EOS:
        if len(sentence) > 0 and sentence[-1] == EOS:
            sentence = sentence[:-1]

    return sentence




def indices_to_words(src_list: List, vocab: _Dict, oov_vocab: List) -> List:
    pred_list = [None] * len(src_list)  # src_list.size(0)
    for ind, src_index in enumerate(src_list):
        p = vocab.lookup_label(src_index)
        if p is None and oov_vocab:
            oov_index = src_index - vocab.size
            try:
                pred_list[ind] = oov_vocab[oov_index]
            except:
                print(oov_index)
                assert False
        else:
            pred_list[ind] = p
    return pred_list

def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).to(dtype=torch.bool), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result
