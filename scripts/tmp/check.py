# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from src import *
from src.data import *

import glob
import random
import ujson
import math

from src.utils.utils import *
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

def parse(log,linenum):
    with open(log,mode='r',encoding='utf-8') as f:
        lines = f.readlines()
        # cnt = 0
        num_list = []
        num_dict = {}
        for i in range(linenum):
            li = lines[i]
            if "batch['id'][b]" in li:
                b = int(li.split("batch['id'][b]:")[-1].split()[0])
                c = int(li.split("sum(non_padding_mask[b]):")[-1].split()[0])
                # assert  cnt == b,print("cnt:{} b:{} \n li: \n{}".format(cnt,b,li ))
                num_dict[b]=c
                # num_list.append(c)
                # cnt+=1
        for i in range(len(num_dict)):
            num_list.append(num_dict[i])
        return num_list

# a="[2020-02-20 12:34:55]    INFO >> batch['id'][b]:233643 sum(non_padding_mask[b]):11  (kd_sl_trainer.py:133, train())"
# b=a.split("batch['id'][b]:")[-1].split()[0]
# c=a.split("sum(non_padding_mask[b]):")[-1].split()[0]

if __name__ == "__main__":
    # 检查 kd 有没有问题

    # python -m scripts.tmp.check
    load_keys = ['comment' ]
    data_lng = 'python'
    file_dir = '/data/wanyao/ghproj_d/naturalcodev2/datasetv2/key/100_small'
    log_dir = '/data/sjd/d/p_d/fse20all/100/log/summarization'
    if data_lng == 'python':
        log = 'SUMMARIZATION_XLANG_KD_MM2SEQ_TRAIN_SL_2020-Feb-20-15-25-33.adf04ea39f56e512eceb9ca89e5ad44e.log' # python
        linenum = 20423
    mode = 'train'

    dataset_files = {
        key: sorted([filename for filename in
                     glob.glob('{}/*'.format(os.path.join(file_dir, data_lng, key)))
                     if mode in filename])
        for key in load_keys
    }

    data = load_data(dataset_files,mode=mode)
    # json_data = load_json(json_path)
    num_list = parse(os.path.join(log_dir,log),linenum)
    print("len(num_list): ",len(num_list))

    for i in range(len(num_list)):
        assert num_list[i] == len(data['comment'][i])+1,print(
            "Assert error, i:{} num_list[i]:{} len(data['comment'][i]):{}".format(i, num_list[i],len(data['comment'][i])))
        # assert num_list[i] == len(data['tok'][i])+1,print(
        #     "Assert error, i:{} num_list[i]:{} len(data['tok'][i]):{}".format(i, num_list[i], len(data['tok'][i])))
    print("check ok for {} ".format(data_lng))