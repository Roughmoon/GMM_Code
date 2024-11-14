import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
from pprint import pprint


class DataSet():
    def __init__(self, dir):
        self.dir = dir
        
    def loadData(self, sheetName="Sheet1"):
        self.df = pd.read_excel(self.dir, sheet_name=sheetName)
        self.df = self.df.dropna()
        # 初始化 StandardScaler
        scaler = StandardScaler()
        # 选择需要标准化的列
        columns_to_scale = self.df.columns[1:]
        # 对选定的列进行标准化
        self.df[columns_to_scale] = scaler.fit_transform(self.df[columns_to_scale])
        # print('数据集加载完成,标准化后为:', self.df)
    def DataToArr(self):
        grouped = self.df.groupby('CompanyCode')
        groupedarrays = [group.values[:, 1:] for _, group in grouped]
        groupedarraysKeys = [i for i, _ in grouped]
        self.groupedarrays = groupedarrays
        self.groupedarraysKeys = groupedarraysKeys

if __name__ == '__main__':
    dataset = DataSet('D:\\PythonClass\\PythonProjects\\GMMCoding\\DataPlus.xlsx')
    dataset.loadData(sheetName="Sheet7")
    dataset.DataToArr()
    