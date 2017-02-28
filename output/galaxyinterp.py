import numpy as np
from scipy.interpolate import interp1d, interp2d

v = np.array([15, 18, 21])

g = []
g.append(np.array([15.13, 18.12, 0]))
g.append(np.array([16.75, 19.7, 0]))
g.append(np.array([18.18, 20.67, 0]))

g3 = []
for i in g:
    m = (i[1] - i[0]) / (v[1] - v[0])
    c = i[1] - m * v[1]
    g3.append([m * 16 + c, i[1], m * 20 + c])

print(g3)