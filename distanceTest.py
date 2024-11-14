import numpy as np
A = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
B = np.array([[1, 2], [2, 3]])
# print(A)
# print(B)
A_expanded = A[:, np.newaxis]
print(A_expanded)
print(A_expanded - B)
C = A_expanded - B
D = np.linalg.norm(C, axis=2)
print(D)
