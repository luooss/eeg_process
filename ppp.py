import numpy as np
from collections import OrderedDict, namedtuple
from itertools import product

params = OrderedDict(lr = np.logspace(1, 2, 2),
                     lambda_ = np.logspace(1, 2, 2),
                     alpha = np.logspace(1, 2, 2))
Run = namedtuple('Run', params.keys())
runs = []
for v in product(*params.values()):
    runs.append(Run(*v))

result = {}
for i in range(8):
    result[runs[i]] = i

print(result[runs[2]])
print(result[runs[5]])
print(result[runs[1]])