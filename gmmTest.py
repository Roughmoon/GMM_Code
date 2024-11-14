import numpy as np
from sklearn.mixture import GaussianMixture
import pandas as pd

class DataSet:
    def __init__(self, dir):
        self.dir = dir
        
    def loadData(self, sheetName='Sheet1'):
        self.df = pd.read_excel(self.dir, sheet_name=sheetName)
        self.df = self.df.dropna()
    
    def DataToArr(self):
        grouped = self.df.groupby('CompanyCode')
        groupedarrays = [group.values[:, 1:] for _, group in grouped]
        groupedarraysKeys = [i for i, _ in grouped]
        self.groupedarrays = groupedarrays
        self.groupedarraysKeys = groupedarraysKeys

# 加载数据
dataset = DataSet('D:\\OneDrive\\3考博大业\\朱映秋老师\\论文代码复现\\Coding\\Data\\DataPlus.xlsx')
dataset.loadData(sheetName='Sheet2')
dataset.DataToArr()

# 假设 M 是一个包含多个 np 数组的列表
M = dataset.groupedarrays

# 将所有 np 数组合并成一个大的 np 数组
all_data = np.concatenate(M, axis=0)

# 定义 GMM 的组件数量
n_components = 4

# 初始化 GMM 并提取初始均值和协方差
gmm = GaussianMixture(
    n_components=n_components, 
    random_state=0, 
    max_iter=0, 
    reg_covar=1e-6,  # 增加正则化项
    init_params='k-means++'  # 使用 k-means++ 初始化
)
gmm.fit(all_data)

# 提取初始均值和协方差
initial_means = gmm.means_
initial_covariances = gmm.covariances_

# 将每个高斯函数的均值和协方差拼接成一个新的数组
combined_arrays = []
for i in range(n_components):
    mean = initial_means[i]
    covariance = initial_covariances[i]
    combined_array = np.hstack((mean.reshape(-1, 1), covariance))
    combined_arrays.append(combined_array)

# 打印每个高斯函数的均值和协方差
for i, combined_array in enumerate(combined_arrays):
    print(f"Gaussian Function {i+1} Combined Array:\n{combined_array}")

# 将所有数组存储在一个列表中
print("All Combined Arrays:")
for i, combined_array in enumerate(combined_arrays):
    print(f"Gaussian Function {i+1} Combined Array:\n{combined_array}")