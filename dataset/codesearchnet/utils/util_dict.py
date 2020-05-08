# -*- coding: utf-8 -*-

from typing import *
import itertools
import ujson

from ncc.utils.constants import *
from ncc.data.dict import Dict as _Dict
from ncc.utils.utils import *


def min_freq_and_sbt_flag(dataset_name: str) -> Tuple:
    '''
    return min_freq and sbt_flag base on dataset_name
    if dataset_name = deepcom/deepcom2, then return min_freq=4, sbt_flag=True
                                        else return min_freq=2, sbt_flag=False
    :param dataset_name:
    :return:
    '''
    min_freq = 2
    if dataset_name == 'deepcom':
        sbt = True  # only True for deepcom dataset
        min_freq *= 2
    else:
        sbt = False
    return min_freq, sbt,


def sort_by_freq(token_freq_dict: Dict) -> List[Tuple]:
    '''
    sort tokens by frequency
    :param token_freq_dict:
    :return:
    '''
    sorted_token_list = sorted([(token, freq) for (token, freq) in token_freq_dict.items()],
                               key=lambda token_freq: -token_freq[-1])
    # token_list, _ = zip(*sorted_token_list)
    return sorted_token_list[:MAX_TOKEN_SIZE]
    # return sorted_token_list


def dump_dict(tokens: List, filename: str, special_token_list=None) -> None:
    '''
    write tokens into {filename}
    :param tokens:
    :param filename:
    :param special_token_list:
    :return:
    '''
    if special_token_list is None:
        special_token_list = [PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD]
    with open(filename, 'w', encoding='utf-8') as writer:
        # write special tokens
        data = []
        for s_token in special_token_list:
            data.append((s_token, POS_INF,))
            # writer.write('{}\t{}\n'.format(s_token, POS_INF))
        for token, freq in tokens:
            # writer.write('{}\t{}\n'.format(token, freq))
            data.append((token, freq,))
        writer.write(ujson.dumps(data))


def write_tokens_dict(code_tokens: List, code_token_filename: str,
                      comment_tokens: List, comment_token_filename: str, ) -> None:
    '''
    save code/comment dicts
    :param code_tokens:
    :param code_token_filename:
    :param comment_tokens:
    :param comment_token_filename:
    :return:
    '''
    # save code dict
    dump_dict(code_tokens, code_token_filename)
    print('write code dict in {}'.format(code_token_filename))

    # save comment dict
    dump_dict(comment_tokens, comment_token_filename)
    print('write code dict in {}'.format(comment_token_filename))


def merge_freqency(multi_dicts: List[Dict]) -> Dict:
    all_dict = dict()
    for dct in multi_dicts:
        for key, value in dct.items():
            if key in all_dict:
                all_dict[key] += value
            else:
                all_dict[key] = value
    return all_dict


def merge_tokens_and_filter(counter_list: List[Counter], min_freq: int) -> List:
    dict_list = [{
        key: vaue
        for key, vaue in counter.items()
    } for counter in counter_list if counter is not None]
    token_dict = merge_freqency(dict_list)

    tokens_dict = {key: freq for key, freq in token_dict.items() if freq > min_freq}
    sorted_tokens = sort_by_freq(tokens_dict)
    return sorted_tokens


def parse_token_dicts(code_counter_list: List[Counter], code_dict_filename: str,
                      comment_counter_list: List[Counter], comment_dict_filename: str,
                      min_freq: int):
    '''
    filter code/comment tokens whose frequency <= min_freq, and save filtered tokens
    :param code_counters:
    :param code_dict_filename:
    :param comment_counters:
    :param comment_dict_filename:
    :param min_freq:
    :return:
    '''
    code_tokens = merge_tokens_and_filter(code_counter_list, min_freq)
    comment_tokens = merge_tokens_and_filter(comment_counter_list, min_freq)
    write_tokens_dict(code_tokens, code_dict_filename,
                      comment_tokens, comment_dict_filename)


def parse_and_write_dicts(counter_list: List, dict_filenames: List, min_freq: int, ) -> None:
    for counter, filename in zip(counter_list, dict_filenames):
        tokens = merge_tokens_and_filter(counter, min_freq)
        # save code dict
        dump_dict(tokens, filename)
        print('write tokens dict in {}'.format(filename))
