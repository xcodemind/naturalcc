import torch
import time
from torch import cuda
import os


def get_num(mem):
    # num = int((mem * 1024 * 1024 / 4) / 0.78369140625 / 0.7527580772261623)
    num = int((mem * 1024 * 1024 / 4) )
    num4_one = int(num / 4)
    num2_one = 2 * num4_one
    return num2_one

# def occupy_gpu(device: int, gb: int, compute=False):
def occupy_gpu(device: int, gb: float, compute=False):
    # set device
    cuda.set_device(device)

    mem_to_use = gb * 1024 - 600
    num2_one = get_num(mem_to_use)

    a = torch.rand(num2_one).reshape(2, -1).to(device)
    # print("a cuda ok")
    b = torch.rand(num2_one).reshape(-1,2).to(device)
    # print("b cuda ok")

    if compute == True:
        print("Mode: gpu computing")
        cnt = 0
        while True:
            time.sleep(1)

            cnt += 1
            if cnt >= 12:
                cnt = 0
                if gb == 0:
                    gb = 0.1

                loop = int(13 /gb)
                # print("loop: ",loop)
                print("start gpu computing device {} gb {}".format(device,gb ))

                for _i in range(loop):
                    c = torch.matmul(a, b)
                # print("computing completed")

    else:
        print("Mode: no gpu computing")
        while True:
            time.sleep(1)

def occupy_gpu_new(device: int, gb: float, compute=False):
    # set device
    cuda.set_device(device)

    mem_to_use = gb * 1024 - 600
    num2_one = get_num(mem_to_use)

    a = torch.rand(num2_one).reshape(2, -1).to(device)
    # print("a cuda ok")
    b = torch.rand(num2_one).reshape(-1,2).to(device)
    # print("b cuda ok")

    if compute == True:
        print("Mode: gpu computing")
        cnt = 0
        while True:
            time.sleep(1)

            cnt += 1
            if cnt >= 12:
                cnt = 0
                if gb == 0:
                    gb = 0.1

                loop = int(13 /gb)
                # print("loop: ",loop)
                print("start gpu computing")

                for _i in range(loop):
                    c = torch.matmul(a, b)
                # print("computing completed")

    else:
        print("Mode: no gpu computing")
        while True:
            time.sleep(1)