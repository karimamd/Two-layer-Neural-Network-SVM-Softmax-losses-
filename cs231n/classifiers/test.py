import numpy as np


Z=[np.array([1.5, 2]),np.array([3, 4]),np.array([7,8])]
c=[x for i,x in enumerate(Z) if i!=2]
print(c)