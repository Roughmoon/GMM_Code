import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
class DataSet():
    def __init__(self, dir):
        self.dir = dir
        
    def loadData(self, sheetName="Sheet1"):
        self.df = pd.read_excel(self.dir,sheet_name=sheetName)
        self.df = self.df.dropna()
    
    def DataToArr(self):
        grouped = self.df.groupby('CompanyCode')
        groupedarrays = [group.values[:,1:] for _, group in grouped]
        groupedarraysKeys = [i for i, _ in grouped]
        self.groupedarrays = groupedarrays
        self.groupedarraysKeys = groupedarraysKeys
        
class Obj():
    def __init__(self, M, G:int, i = 0, sum=10):
        self.M = M
        self.G = G
        self.dim = M[0].shape[1]
        self.n = len(M)
        self.i = i
        self.sum = sum
    def generate_random_multivariate_normal(self, seed=None):
        """
        随机生成多维正态分布的均值向量和协方差矩阵

        参数:
        dim: 正态分布的维度
        seed: 随机种子，用于复现相同的随机数

        返回:
        mu: 均值向量
        cov: 协方差矩阵
        """
        if seed is not None:
            np.random.seed(seed)
        # 生成均值向量
        mu = np.random.randn(self.dim)*10
        # 生成协方差矩阵
        A = np.random.randn(self.dim, self.dim)*10
        cov = A @ A.T  # 确保协方差矩阵是对称正定的
        return mu, cov

    def initGMM(self):
        # GMM = []
        GMM = [np.array([[0,1,0.5],[0,0.5,1]]),np.array([[0,2,1],[0,1,2]])]
        for _ in range(self.G):
            mu, cov = self.generate_random_multivariate_normal()
            GMM.append(np.hstack((mu[:, np.newaxis], cov)))
        # print('Initial GMM:', GMM)  # 打印初始化后的 GMM
        return GMM
    
    def initPHI(self):
        PHI = [np.full(self.G, 1.0/self.G) for _ in range(self.n)]
        return PHI
    def muldiNorFunc(self, X, par:np.array)->float:
        """
            计算多元高斯分布的概率密度函数值
            参数：
                dim: 维度
                X: 样本点
                par: 参数，包含均值和协方差
            返回值：
                概率密度函数值
        """
        mu = par[:,0]
        var = par[:,1:]
        # 确保协方差矩阵中没有无效值
        if not np.isfinite(var).all():
            raise ValueError("协方差矩阵中包含无效值（如inf或NaN）")
        # 检查协方差矩阵的对称性
        if not np.allclose(var, var.T):
            raise ValueError("协方差矩阵必须是对称的")

        # 检查协方差矩阵的正定性
        eigenvalues = np.linalg.eigvals(var)
        if np.any(eigenvalues <= 0):
            raise ValueError("协方差矩阵必须是正定的")
        dist = multivariate_normal(mean=mu, cov=var)
        value = dist.pdf(X)
        return value
    
    def develop(self, GMM, PHI):
        self.KSI = []
        def EStep():
            for i in range(self.n):
                ksi = np.zeros((self.M[i].shape[0], self.G))
                for j in range(self.M[i].shape[0]):
                    f_g = np.array([self.muldiNorFunc(self.M[i][j], GMM[g]) for g in range(self.G)])
                    # print('第{}个样本点的第{}个分量的概率'.format(i+1, j+1), f_g,'\n')
                    denominator = np.dot(f_g, PHI[i])
                    # print('denominator', denominator)
                    ksi_j = (f_g * PHI[i])/denominator
                    # print('第{}个样本点的第{}个分量的概率'.format(i+1, j+1), ksi_j,'\n')
                    ksi[j] = ksi_j
                self.KSI.append(ksi)
            # print('KSI的更新值', self.KSI)
                
            for i in range(self.n):
                for g in range(self.G):
                    PHI[i][g] = np.sum(self.KSI[i][:,g])/self.M[i].shape[0]
            # print('Phi的更新值', PHI)
        def MStep():
            for g in range(self.G):
                mu = np.zeros(self.dim)
                ksi_sum_g = 0
                for i in range(self.n):
                    ksi_sum_g += float(np.sum(self.KSI[i][:,g]))
                for i in range(self.n):
                    for j in range(self.M[i].shape[0]):
                        mu += self.KSI[i][j,g]*self.M[i][j]
                mu /= ksi_sum_g
                # print('第{}个高斯分布的均值向量为:'.format(g+1), mu)
                GMM[g][:,0] = mu
            
            for g in range(self.G):
                cov = np.zeros((self.dim, self.dim))
                ksi_sum_g = 0
                for i in range(self.n):
                    ksi_sum_g += float(np.sum(self.KSI[i][:,g]))
                for i in range(self.n):
                    for j in range(self.M[i].shape[0]):
                        Mij_minus_mu = self.M[i][j] - self.GMM[g][:,0]
                        Mij_minus_mu = Mij_minus_mu.reshape(-1,1)
                        cov += self.KSI[i][j,g]*(Mij_minus_mu @ Mij_minus_mu.T)
                cov /= ksi_sum_g
                print('第{}个高斯分布的协方差矩阵为:'.format(g+1), cov)
                GMM[g][:,1:] = cov
        EStep()
        MStep()

    def EM(self, T0):
        self.GMM = self.initGMM()
        self.PHI = self.initPHI()
        # for t in tqdm(range(T0), desc="EM Iterations"):
        for t in range(T0):
            self.develop(self.GMM, self.PHI)
            print('Phi的更新值为:', self.PHI)
        return self.PHI, self.GMM
    
if __name__ == '__main__':
    dataset = DataSet('D:\\OneDrive\\3考博大业\\朱映秋老师\\论文代码复现\\Coding\\Data\\DataPlus.xlsx')
    dataset.loadData(sheetName="Sheet6")
    dataset.DataToArr()
    obj = Obj(M = dataset.groupedarrays, G = 5)
    Phi, Gmm = obj.EM(100)
    


