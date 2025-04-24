import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 2, 23, 4, 1])

jumps = len(a)-np.sum(a == b)
print( 'sum', np.sum(a == b))
print('time jumps', jumps,  '\n')