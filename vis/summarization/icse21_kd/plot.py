# -*- coding: utf-8 -*-


"""
"#cc79a7", # 紫红
"#e69f00" ,# 橙色
"#f0e442" ,# 亮黄
"#d55e00" # 朱红
"""

import os
import ujson
import itertools
import math
import matplotlib.pyplot as plt

from dataset.codesearchnet.utils.constants import LANGUAGES, MODES

plt.figure()
plt.hist()
plt.xlabel('length of {}.{}.{}'.format(lang, mode, modality))
plt.ylabel('frequency')
# plt.show()
plt.savefig(file, transparent=True)
plt.close()
