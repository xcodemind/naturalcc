# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

from typing import *

import os
import ujson
import itertools

PRED_DIR = '/data/wy/ghproj_d/fse20all/100_small/casestudypred_202034075103'
pred_files = {
    '0': {
        'seq2seq': 'seq2seqv2_p0.pred',
        'code2seq': 'code2seq_p0.pred',
        'mm2seq': 'tok8pathattnpointer_p0.pred',
        'ft_java': 'ft_java_pe1_p0.pred',
        'ft_js': 'ft_java_pe1_p0.pred',
        'ft_php': 'ft_php_pe1_p0.pred',
        'ft_python': 'ft_python_pe1_p0.pred',
        'maml': 'maml_p0.pred',
    },

    '1': {
        'seq2seq': 'seq2seqv2_p1.pred',
        'code2seq': 'code2seq_p1.pred',
        'mm2seq': 'tok8pathattnpointer_p1.pred',
        'ft_java': 'ft_java_pe1_p1.pred',
        'ft_js': 'ft_java_pe1_p1.pred',
        'ft_php': 'ft_php_pe1_p1.pred',
        'ft_python': 'ft_python_pe1_p1.pred',
        'maml': 'maml_p1.pred',
    },

}
for portion, p_files in pred_files.items():
    for model, pt_file in p_files.items():
        pred_files[portion][model] = os.path.join(PRED_DIR, pt_file)

print(pred_files)


def load_a_line(readers):
    info = {}
    for model, reader in readers.items():
        line = reader.readline().strip()
        if len(line) > 0:
            info[model] = ujson.loads(line)
        else:
            return {}
    return info


def filter(readers):
    output = []
    while True:
        info = load_a_line(readers)
        if len(info) > 0:
            if info['maml']['cider'] >= 2.0:
                output.append(info)
        else:
            return output


def csv_line(infos, ):
    all_models = sorted(list(infos.keys()))
    line = [infos['maml']['src_code'], str.lower(infos['maml']['src_comment']), \
            str.lower(' '.join(infos['maml']['trg_comment'])), ]
    for model in all_models:
        line.extend([str.lower(' '.join(infos[model]['pred_comment'])), infos[model]['cider'], ])
    return line


import csv

for portion, p_files in pred_files.items():
    p_readers = {model: open(pt_file, 'r') for model, pt_file in p_files.items()}
    p_infos = filter(p_readers)
    p_infos = sorted(p_infos, key=lambda info_list: -info_list['maml']['cider'])

    trg_file = open('/data/wanyao/yang/ghproj_d/GitHub/naturalcodev2/case_study_p{}.csv'.format(portion), 'w')
    csv_writer = csv.writer(trg_file)

    all_models = sorted(list(p_infos[0].keys()))
    head = ['src_code', 'src_comment', 'trg_comment', ] + \
           list(itertools.chain(*[[model, 'cider'] for model in all_models]))
    csv_writer.writerow(head)

    for infos in p_infos:
        line = csv_line(infos)
        csv_writer.writerow(line)
        csv_writer.writerow([])


def main():
    pass


if __name__ == '__main__':
    main()
