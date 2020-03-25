# -*- coding: utf-8 -*-

import sys

sys.path.append('./')

from run.util import *
from ncc.utils.util_gpu import occupy_gpu_new
from ncc.utils.util_file import load_yaml
import time
import subprocess

if __name__ == '__main__':
    arg = get_args_new()
    # print(args)
    # assert False
    # while True:

    # args = DictAsObj(load_yaml(arg.yml))
    opt = load_yaml(os.path.join(os.path.dirname(__file__), arg.yml))
    # print("opt: ",opt )
    args = DictAsObj(opt)
    # print("args: ",args)

    log_filename = get_log_filename(args)
    while os.path.exists(log_filename):
        time.sleep(1)
        log_filename = get_log_filename(args)
    # print(log_filename)


    # save_dir = '/data/wanyao/ghproj_d/naturalcodev2/datasetv2/result'

    log_path =  os.path.join(args.dir,'log', args.task )
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    save_dir = os.path.join(args.dir,'result')
    del args.dir
    args.save_dir = save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # print("before appendix , args: ",args )
    appendix = ' '.join(['--{} {}'.format(key, value) for key, value in args.items()])

    # command = 'python -u {} {} > {}'.format(
    #     os.path.join(sys.path[0], args.task, args.lang_mode, args.method_name, 'main.py', ),
    #     appendix,
    #     os.path.join(log_path, log_filename, )
    # )
    # command = 'nohup python -u {} {} > {}'.format(
    #     os.path.join(sys.path[0], args.task, args.lang_mode, args.method_name, 'main.py', ),
    #     appendix,
    #     os.path.join(log_path, log_filename )
    # )

    command = 'nohup python -u {} {} | tee {}'.format(
        os.path.join(sys.path[0], args.task, args.lang_mode, args.method_name, 'new_main.py', ),
        appendix,
        os.path.join(log_path, log_filename )
    )

    # command = 'CUDA_LAUNCH_BLOCKING=1 nohup  python -u {} {} | tee {}'.format(
    #     os.path.join(sys.path[0], args.task, args.lang_mode, args.method_name, 'new_main.py', ),
    #     appendix,
    #     os.path.join(log_path, log_filename )
    # )

    try:
        print(command)

        run_code = os.system(command)
        # run_code, run_output = subprocess.getstatusoutput(command)
        if run_code == 0:
            print("finished.")
        else:
            print("failed.")
    except Exception as err:
        print(err)

    if args.occupy_gpu != 'no':
        print('Occupying GPU. Enter Ctrl+C to complete.')
        # gb = args.occupy_gpu
        # device = int(device[1:])
        # device = args.device
        # gb = int(gb[:-2])
        gb = float(args.occupy_gpu)
        print("try occupy_gpu ,device:{} gb:{}".format(args.device,gb ))
        # occupy_gpu_new(args.device, gb, compute=False)
        occupy_gpu_new(args.device, gb, compute=True )
