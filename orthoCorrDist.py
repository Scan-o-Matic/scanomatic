import numpy as np
V = np.array([[1, 1], [2, 1], [1, 2], [4, 1], [1, 4]], dtype=np.float)
u = np.array((1, 0), dtype=np.float)
uHat = u / np.sqrt((u ** 2).sum())
O = np.outer(np.apply_along_axis(np.dot, 1, V, uHat), uHat)
L = np.sqrt((O ** 2).sum(axis=1))
for v, o, l in zip(V, O, L):
    print "{0} => {1} ({2})".format(v, o, l)
