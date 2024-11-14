import numpy as np

cluster_labels = np.array([0, 0, 1, 1, 2, 2])
data = np.array([[1,2], [3,4], [5,6], [7,8], [9,10], [11,12]])
data0 = data[cluster_labels ==0]
data1 = data[cluster_labels ==1]
data2 = data[cluster_labels ==2]
# data2 = np.array([data[cluster_labels ==i] for i in range(6)])

# new_centroids = np.array([data[cluster_labels == i].mean(axis=0) for i in range(k)])