import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成100个三维向量 (x, y, z)
np.random.seed(0)  # 为了可重复性
vectors = np.random.rand(100, 3)

# 提取 x, y, z 值
x = vectors[:, 0]
y = vectors[:, 1]
z = vectors[:, 2]

# 创建三维图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面图
surf = ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')

# 添加颜色条
fig.colorbar(surf, shrink=0.5, aspect=5)

# 设置标签
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# 设置标题
ax.set_title('3D Surface Plot of 100 Vectors')

# 显示图形
plt.show()