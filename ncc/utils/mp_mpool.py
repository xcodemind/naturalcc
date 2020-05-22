# -*- coding: utf-8 -*-

from typing import *

import multiprocessing as mp


class MPool:
    """
    design for NCC dataset generation
    """

    def __init__(self, processor_num: int = None, ):
        self.processor_num = 1 if processor_num is None \
            else min(processor_num, mp.cpu_count())

    def feed(self, func: Any, params: List) -> List[Any]:
        data_buffer = mp.Queue()
        processor_buffer = []

        out = []
        for p_id in range(len(params)):
            tmp_p = mp.Process(target=self._fun_with_id, args=(func, params[p_id], p_id, data_buffer,))
            tmp_p.daemon = True
            tmp_p.start()
            processor_buffer.append(tmp_p)
            if len(processor_buffer) == self.processor_num or p_id == len(params) - 1:
                for p in processor_buffer:
                    p.join()
                for p in processor_buffer:
                    p.terminate()
                    p.join()
                processor_buffer = []

                result = []
                while not data_buffer.empty():
                    result.append(data_buffer.get())
                assert data_buffer.empty() == True
                out.extend(result)

        out = sorted(out, key=lambda res_id: res_id[-1])
        out, _ = zip(*out)
        data_buffer.close()
        del processor_buffer, data_buffer
        return list(out)

    def _fun_with_id(self, func: Any, params: List, id: int, buffer: mp.Queue):
        result = func(*params)
        buffer.put((result, id))
