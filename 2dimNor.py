import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

# 定义均值和协方差矩阵
mean = np.array([0, 0])  # 均值向量
cov = np.array([[2, 1], [1, 2]])  # 协方差矩阵

# 生成网格点
x, y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
pos = np.dstack((x, y))

# 创建多维正态分布对象
rv = multivariate_normal(mean, cov)

# 计算概率密度函数值
z = rv.pdf(pos)

# 绘制图像
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制三维曲面
ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')

# 绘制均值点
ax.scatter(mean[0], mean[1], 0, color='red', marker='x', s=100, label='Mean')

# 设置标题和标签
ax.set_title('3D Gaussian Distribution')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Probability Density')

# 显示图例
ax.legend()

# 显示图像
plt.show()