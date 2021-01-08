"""=============================================================================
Ex2: EIGENDECOMPOSITION
    Câu 1:
        a) Tạo ma trận A(5 x 5) chứa các giá trị ngẫu nhiên trong khoảng 1 - 10
        b) Phân tích eigenvalues và eigenvectors
        c) Kiểm tra eigenvector đầu tiên theo dot và theo eigenvalue có bằng nhau?
           Nếu bằng nhau thì tái tạo A từ các eigenvalues và eigenvectors
============================================================================="""
    
import numpy  as np
import random       

from numpy.linalg import eig, inv
from numpy import diag

##------------------------------------------------------------------------------
## Hàm tạo 1 ma trận A[mxn] với giá trị ngẫu nhiên thuộc [start, end]
##------------------------------------------------------------------------------
def create_matrix_random(m, n, start, end):
    mtr = []
    for i in range(m):
        row = []
        for j in range(n):
            a = random.randint(start, end + 1)
            
            # Thêm giá trị vào dòng hiện hành 
            row.append(a)
            
        # Thêm dòng vào ma trận    
        mtr.append(row)
        
    return np.array(mtr)
##------------------------------------------------------------------------------
    
## a) Tạo ma trận A(5 x 5) chứa các giá trị ngẫu nhiên trong khoảng 1 - 10
m, n, min, max = 5, 5, 1, 10
A = create_matrix_random(m, n, min, max)
print('Ma trận A', A.shape, ':\n', A)

## b) Phân tích eigenvalues và eigenvectors
values, vectors = eig(A)
print('\nSố lượng giá trị riêng: %d' %len(values))
print('\nEigenvalues',  values.shape,  ':\n', values)
print('\nEigenvectors', vectors.shape, ':\n', vectors)

## c) Kiểm tra eigenvector đầu tiên theo dot và theo eigenvalue có bằng nhau?
##    Nếu bằng nhau thì tái tạo A từ các eigenvalues và eigenvectors
B = A.dot(vectors[:, 0])
print('\nMa trận B:\n', B.astype(int))

C = vectors[:, 0] * values[0]
print('\nMa trận C:\n', C.astype(int))

Q = vectors
L = diag(values)

print('\nTái tạo ma trận A:\n', Q.dot(L).dot(inv(Q)).astype(int))
