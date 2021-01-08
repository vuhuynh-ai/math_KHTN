import numpy as np
import random
from scipy.linalg import lu
A = np.array([[2, 5, 8, 7, 0], [5, 2, 2, 8, 0], [7, 5, 6, 6, 0], [5, 4, 4, 8, 0], [2, 4, 1, 0, 0]])
p, l, u = lu(A)
print(p)
print(l)
print(u)
a = p.dot(l).dot(u)
print(a)

def new_matrix(m, n, start, end):
    mat = []
    for i in range(m):
        row = []
        for j in range(n):
            x = random.randint(start, end+1)
            row.append(x)
        mat.append(row)
    return np.array(mat)
Z = new_matrix(3,4,1,10)
print('matran k vuong:',Z.shape, ':\n', Z)
