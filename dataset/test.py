# -*- coding: utf-8 -*-

from typing import *

from multiprocessing import Process, Queue


class Test:

    def __init__(self):
        self.thread_num = 3
        self.tmp = TMP()

    def parallel(self):
        q = Queue()

        process_list = []
        for i in range(self.thread_num):  # 开启5个子进程执行fun1函数
            p = Process(target=self.tmp.func, args=(q, 'Python_{}'.format(i),))  # 实例化进程对象
            p.daemon = True
            p.start()
            process_list.append(p)

        for p in process_list:
            p.join()
        for p in process_list:
            p.terminate()
            p.join()
        while not q.empty():
            print(q.get())
        print('结束测试')

    def fun1(self, q: Queue, name):
        # print('测试%s多进程' % name)
        q.put(name)


class TMP:

    def func(self, q: Queue, param, ):
        q.put(param)


if __name__ == '__main__':
    # test = Test()
    # test.parallel()
    a = ['A', 'b']
    print(list(map(str.lower, a)))
