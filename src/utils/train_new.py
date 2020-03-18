import torch
import time
from torch import cuda
import argparse
import os
# import random
# # import pynvml
# import threading
# import gc
# pip install nvidia-ml-py3


def get_num(mem):
    # num = int((mem * 1024 * 1024 / 4) / 0.78369140625 / 0.7527580772261623)
    num = int((mem * 1024 * 1024 / 4) )
    return num

def get_num2(mem):
    # num = int((mem * 1024 * 1024 / 4) / 0.78369140625 / 0.7527580772261623)
    num = int((mem * 1024 * 1024 / 4) )
    num4_one = int(num / 4)
    num2_one = 2 * num4_one
    return num2_one

def get_opt():
    # python -m src.utils.train_new
    parser = argparse.ArgumentParser(description='a')
    parser.add_argument('--gpuid',"-g",type=int,default= 2    )
    parser.add_argument('--gb',"-b", default= 5.4   , type=float)
    parser.add_argument('--compute', "-c", default=False, action ="store_true" )
    # parser.add_argument('--compute', "-c", default=True )

    opt = parser.parse_args()

    return opt

opt = get_opt()
cuda.set_device(opt.gpuid)



gb = opt.gb
mem_to_use = gb*1024 - 600


# num_two = get_num(mem_to_use)
# print("pre d")
# add = torch.rand(num_two)

num2_one = get_num2(mem_to_use)

print("pre a")
a=torch.rand(num2_one).reshape(2,-1)
print("a cpu ok")
a = a.cuda()
# a.requires_grad_()
print("a cuda ok")
print("pre b")
b=torch.rand(num2_one).reshape(-1,2)
print("b cpu ok")
b = b.cuda()
print("b cuda ok")
# # print("d ok")
# add = add.cuda()
# print("gpu ok")
if opt.compute == True:
    print("Mode: gpu computing")
    cnt = 0
    while True:
        print("GPU memory: ", opt.gb)
        time.sleep(1)
        print("pid: ", os.getpid())
        print("GPU: ",opt.gpuid)
        cnt += 1
        if cnt >= 12:
            cnt = 0
            if opt.gb == 0:
                opt.gb = 0.1

            loop = int(13 /opt.gb)
            print("loop: ",loop)
            print("start gpu computing")

            for _i in range(loop):
                c = torch.matmul(a, b)
            print("computing completed")

else:
    print("Mode: no gpu computing")
    while True:
        print("GPU memory: ", opt.gb)
        print("pid: ",os.getpid())
        print("GPU: ", opt.gpuid)
        time.sleep(1)
