# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

from src import *
import glob
import ujson

if __name__ == '__main__':
    # python -m scripts.moses.get_portion_data
    random.seed(666)
    # src_dir = '/data/wanyao/yang/ghproj_d/GitHub/datasetv2/key/100_small/ruby/'
    src_dir = '/data/wanyao/ghproj_d/naturalcodev2/datasetv2/key/100_small/ruby/'

    KEYS = ['tok', 'comment']

    for portion in [0.01,0.2]:
        random_indices = None
        for key in KEYS:
            trg_dir = '/data/wanyao/ghproj_d/naturalcodev2/datasetv2/key/100_small/ruby_p{}/{}'.format(portion, key)
            os.makedirs(trg_dir, exist_ok=True)
            for mode in ['train', 'valid', 'test']:
                files = sorted([fl for fl in glob.glob('{}/*'.format(os.path.join(src_dir, key))) if mode in fl])
                if mode == 'train':
                    # read all train files
                    data = []
                    for fl in files:
                        with open(fl, 'r') as reader:
                            data.extend([line.strip() for line in reader.readlines() if len(line.strip()) > 0])
                    # slice
                    if random_indices is None:
                        random_indices = random.sample(range(len(data)), int(len(data) * portion))
                    print("portion:{} key:{} len(data):{}  len(random_indices):{} ".\
                          format(portion, key, len(data),len(random_indices)))
                    data = [data[ind] for ind in random_indices]
                    # write
                    with open(os.path.join(trg_dir, 'ruby_{}_0.txt'.format(mode)), 'w') as writer:
                        for line in data:
                            writer.writelines(line + '\n')

                else:
                    for fl in files:
                        os.system('cp {} {}'.format(fl, trg_dir))
