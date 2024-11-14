import numpy as np


distances = np.array([[1, 2], [3, 4], [5, 6 ]])
A = np.argmin(distances, axis=0)
B = np.argmin(distances, axis=1)