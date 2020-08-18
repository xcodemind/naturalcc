from typing import Dict, Optional, Set, List
import os
import re
import ujson
import itertools
from ncc.utils import tokenizer
from ncc.data import constants
import ncc.utils.mp_mreader as mreader


def path_special_symbols(files: List[str]) -> Set:
    """get special symbols of path for bert bpe encoding"""
    special_symbols = set()

    def path_body_tokens(line: str, *args, **kwargs):
        line = ujson.loads(line.strip())
        paths = line.split(constants.S_SEP)
        body = [tokenizer.tokenize_string(
            re.split(r'{}|{}'.format(constants.H_SEP, constants.T_SEP), path)[1]
        ) for path in paths]
        return set(itertools.chain(*body))

    for file in files:
        print(file)
        mtokens = mreader.readline(file, path_body_tokens)
        for tokens in itertools.chain(*mtokens):
            special_symbols.update(tokens)
    return special_symbols

if __name__ == '__main__':

    def _special_symbols(args: Dict, modality: str) -> Optional[Set]:
        special_symbols = set()
        if modality in ['code']:
            special_symbols.update([constants.S_SEP, constants.CLS])
        elif modality in ['path']:
            special_symbols.update([constants.H_SEP, constants.T_SEP, constants.S_SEP])
            special_symbols.update(path_special_symbols(args.input_files[modality]))
        else:
            return None
        return special_symbols

    def _get_special_symbols(args: Dict, modality: str) -> Optional[Set]:
        default_special_symbols_files = os.path.join(args.tgt_dir, '.{}.ss'.format(modality))
        if os.path.exists(default_special_symbols_files) and (not args.overwrite):
            with open(default_special_symbols_files, 'r') as reader:
                special_symbols = [line.rstrip('\n') for line in reader.readlines()]
        else:
            special_symbols = _special_symbols(args, modality)
            if special_symbols is None:
                return special_symbols
            os.makedirs(os.path.dirname(default_special_symbols_files), exist_ok=True)
            with open(default_special_symbols_files, 'w') as writer:
                for symbol in special_symbols:
                    writer.write(symbol + '\n')
        return special_symbols

    special_symbols = set()
    # for modality in args.modalities:
    #     sp_symbols = _get_special_symbols(args, modality)
    #     if sp_symbols is None:
    #         pass
    #     else:
    #         special_symbols.update(sp_symbols)
    # return special_symbols