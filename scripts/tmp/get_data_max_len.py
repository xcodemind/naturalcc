# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from src import *
from src.data import *

import glob
import random
import ujson
import math

from src.utils.util import *
from src.utils.util_data import *
from src.utils.util_file import *
from src.data import _Dict
from src.utils.util_data import build_graph

import json
def load_json(path):
    with open(path, 'r', encoding='utf-8') as json_file:
        aa = json_file.readlines()[0]
        output = json.loads(aa)
    return output



def load_file(filename: str) -> List:
    with open(filename, 'r') as reader:
        return [ujson.loads(line.strip()) for line in reader.readlines() if len(line.strip()) > 0]


def load_data(dataset_files: Dict,mode:str ) -> Dict:
    data = {key: [] for key in dataset_files.keys()}
    for key, filenames in dataset_files.items():
        for fl in filenames:
            LOGGER.info("mode:{} keys: {}/{} file: {}/{}".format(mode , key,dataset_files.keys(),filenames.index(fl),len(filenames)))
            data[key].extend(load_file(fl))
    return data


if __name__ == "__main__":
    # 检查 kd 有没有问题

    # python -m scripts.tmp.get_data_max_len
    load_keys = ['comment','tok','sbtao' ]
    data_lng = 'ruby'
    file_dir = '/data/wanyao/ghproj_d/naturalcodev2/datasetv2/key/100_small'

    # mode = 'train'
    maxlen={'tok':0,'sbtao':0,'comment':0}

    for mode in ['train','valid','test']:
        dataset_files = {
            key: sorted([filename for filename in
                         glob.glob('{}/*'.format(os.path.join(file_dir, data_lng, key)))
                         if mode in filename])
            for key in load_keys
        }

        data = load_data(dataset_files,mode=mode)

        for datakey in load_keys:
            print("{} {}".format(mode,datakey))
            for i in range(len(data[datakey])):
                if len(data[datakey][i]) > maxlen[datakey]:
                    maxlen[datakey] = len(data[datakey][i])

    print("maxlen['tok']:{} maxlen['sbtao']: {} maxlen['comment']: {} ".\
          format(maxlen['tok'],maxlen['sbtao'],maxlen['comment']))