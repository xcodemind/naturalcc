# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

import os
import ujson
import matplotlib.pyplot as plt
from pylab import *
import itertools

# DICTS_DIR = r'C:\Users\GS65_2070mq\Documents\GitHub\naturalcodev2\exp\basic_info\100_small'
DICTS_DIR = r'C:\Users\GS65_2070mq\Documents\GitHub\naturalcodev2\exp\basic_info\base'
MODELITIES = ['Tok', 'AST', 'Comment']
LANUAGES = ['Ruby', 'Python', 'PHP', 'Go', 'Java', 'Javascript']

data_files = {
    modal: os.path.join(DICTS_DIR, '{}_size.dict'.format(str.lower(modal)))
    for modal in MODELITIES
}
print(data_files)


def load_dict(filename):
    with open(filename, 'r') as reader:
        return ujson.loads(reader.read())


def only_train(dictionary):
    return {
        str.lower(lng): dictionary[str.lower(lng)]['train']
        for lng in LANUAGES
    }


data = {
    modal: only_train(load_dict(data_files[modal]))
    for modal in MODELITIES
}

fig = plt.figure(figsize=(15, 10))
for lng_ind, lng in enumerate(LANUAGES):
    for modal_ind, modal in enumerate(MODELITIES):
        plt_tmp = fig.add_subplot(len(LANUAGES), len(MODELITIES), lng_ind * len(MODELITIES) + (modal_ind + 1))
        data_tmp = data[modal][str.lower(lng)]
        len_size = len(data_tmp)
        data_tmp = list(itertools.chain(*[[int(length)] * count for length, count in data_tmp.items()]))
        data_tmp = sorted(data_tmp)
        plt_tmp.hist(data_tmp, len_size)

        if modal == 'TOK':
            modal = 'Code'
        plt.title(r'{}-{}'.format(lng, modal), fontdict={'weight': 'normal', 'size': 14})
        # plt.title(r'{}-{}'.format(lng, modal))
fig.tight_layout()
fig.savefig(fname='dist.pdf')
fig.show()
