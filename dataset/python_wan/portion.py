import os
import shutil
import glob
import ujson
from dataset.python_wan import (FILTER_DIR, LOGGER)


def file_line_num(filename):
    with open(filename) as reader:
        return sum(1 for line in reader)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Portion")
    parser.add_argument(
        "--portion", "-p", type=float, nargs='+',
        default=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="portion of train dataset",
    )
    parser.add_argument(
        "--language", "-l", default='python', type=str, help="set constant for language",
    )
    args = parser.parse_args()

    _FILTER_DIR = os.path.expanduser(FILTER_DIR)
    FILE_DIR = os.path.join(_FILTER_DIR, args.language)
    DATA_NUM = file_line_num(os.path.join(FILE_DIR, 'train.code'))

    for prtn in args.portion:
        PORTION_DATA_NUM = int(DATA_NUM * prtn)
        dest_dir = os.path.join(_FILTER_DIR, '{}-{}'.format(args.language, prtn))
        os.makedirs(dest_dir, exist_ok=True)
        LOGGER.info('Partialize {} portion data at {}'.format(prtn, dest_dir))

        for filename in glob.glob(os.path.join(FILE_DIR, '*')):
            if str.startswith(os.path.basename(filename), 'train'):
                if str.endswith(filename, 'path'):
                    dst_data_num = PORTION_DATA_NUM * 300
                elif str.endswith(filename, 'path.terminals'):
                    dst_data_num = PORTION_DATA_NUM * 300 * 2
                else:
                    dst_data_num = PORTION_DATA_NUM

                with open(filename, 'r') as reader, \
                    open(os.path.join(dest_dir, os.path.basename(filename)), 'w') as writer:
                    for idx, src_line in enumerate(reader):
                        if idx < dst_data_num:
                            print(src_line.rstrip('\n'), file=writer)
                        else:
                            break
            else:
                shutil.copy(filename, dest_dir)
