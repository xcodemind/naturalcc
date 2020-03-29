# -*- coding: utf-8 -*-

import sys

sys.path.append('./')

from run.util import *
from ncc.utils.util_gpu import occupy_gpu
import time

def main():
    args = get_args()
    # print(args)
    # assert False
    # while True:
    log_filename = get_log_filename(args)
    while os.path.exists(log_filename):
        time.sleep(1)
        log_filename = get_log_filename(args)
    # print(log_filename)

    appendix = ' '.join(['--{} {}'.format(key, value) for key, value in args.__dict__.items()])

    if args.log_root_dir is not None:
        log_root_dir = args.log_root_dir
        log_path = os.path.join(log_root_dir, 'log', args.task)
    else:
        log_root_dir = '/data/wanyao/ghproj_d/naturalcodev2/datasetv2/result'
        log_path = os.path.join(log_root_dir, args.task, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    '''
    1. python -u *.py
    '''
    # command = 'python -u {} {}'.format(
    #     os.path.join(sys.path[0], args.task, args.lang_mode, args.method_name, 'main.py', ),
    #     appendix,
    # )

    '''
    2. nohup python -u *.py > *.log 
    '''
    # for codebert
    appendix = ' '.join(['--{} {}'.format(key, value) for key, value in {'yaml': args.yaml}.items()])
    command = 'nohup python -u {} {} > {}'.format(
        os.path.join(sys.path[0], args.task, args.lang_mode, args.method_name, 'train_wy.py', ),
        appendix,
        os.path.join(log_path, log_filename, )
    )


    '''
    3. nohup python -u *.py | tee *.log 
    '''
    # command = 'nohup python -u {} {} | tee {}'.format(
    #     os.path.join(sys.path[0], args.task, args.lang_mode, args.method_name, 'main.py', ),
    #     appendix,
    #     os.path.join(log_path, log_filename, )
    # )

    '''
    4. nohup python -u *.py > *.log 2>&1 &
    '''
    # command = 'nohup python -u {} {} > {} 2>&1 &'.format(
    #     os.path.join(sys.path[0], args.task, args.lang_mode, args.method_name, 'main.py', ),
    #     appendix,
    #     os.path.join(log_dir, args.task, 'log', log_filename, )
    # )

    try:
        # print('xxxxx')
        print(command)
        # sys.exit()
        run_code = os.system(command)
        if run_code == 0:
            print("finished.")
        else:
            print("failed.")
    except Exception as err:
        print(err)

    if args.occupy_gpu != 'no':
        device, gb = args.occupy_gpu.split('-')
        # device = int(device[1:])
        device = int(device)
        # gb = int(gb[:-2])
        gb = float(gb)
        print('Occupying GPU. Enter Ctrl+C to complete. gpu: {} gb: {}'.format(device, gb))
        occupy_gpu(device, gb, compute=False)


if __name__ == '__main__':
    main()