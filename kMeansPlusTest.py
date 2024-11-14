# 导入所需库
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd
from sklearn.decomposition import PCA

#我的数据

data = pd.read_excel("D:\\OneDrive\\3考博大业\朱映秋老师\\论文代码复现\\Coding\\Data\\ProPlus.xlsx")
def parse_floats(s):
    # 去除字符串两端的方括号，然后使用空格分割字符串，并将每个部分转换为浮点数
    return list(map(float, s[1:-1].split()))

# 应用这个函数到 'Phis' 列的每一个元素，并将结果转换为NumPy数组
data['Phis'] = data['Phis'].apply(parse_floats)

# 由于每个元素现在都是一个列表，我们可以使用列表推导式结合NumPy来创建一个二维数组
phis_2d_array = np.array([np.array(x) for x in data['Phis']])
# 使用KMeans++算法进行聚类
kmeans_plus = KMeans(n_clusters=10, init='k-means++') # 'k-means++' 是关键参数
kmeans_plus.fit(phis_2d_array)

# 获取聚类标签
cluster_labels = kmeans_plus.labels_

# 获取质心
centroids = kmeans_plus.cluster_centers_

# 打印聚类标签和质心
print("聚类标签:", cluster_labels)
print("质心:", centroids)
print("聚类数量：", len(cluster_labels))


# # 可视化聚类结果
# # 如果数据是高维的，先进行降维
# if phis_2d_array.shape[1] > 2:
#     pca = PCA(n_components=2)
#     reduced_data = pca.fit_transform(phis_2d_array)
# else:
#     reduced_data = phis_2d_array

# # 绘制散点图
# plt.figure(figsize=(10, 6))
# for i in range(10):
#     plt.scatter(reduced_data[cluster_labels == i, 0], reduced_data[cluster_labels == i, 1], label=f'Cluster {i}')

# # 绘制质心
# plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')

# plt.title('K-means Clustering')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.legend()
# plt.show()