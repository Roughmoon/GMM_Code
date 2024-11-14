import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import math
class DataSet():
    def __init__(self, dir):
        self.dir = dir
        
    def loadData(self):
        self.df = pd.read_excel(self.dir)
        # print(self.df.head())
    
    def DataToArr(self):
        grouped = self.df.groupby('CompanyCode')
        groupedarrays = [group.values[:,1:] for _, group in grouped]
        groupedarraysKeys = [i for i, _ in grouped]
        self.groupedarrays = groupedarrays
        self.groupedarraysKeys = groupedarraysKeys
        # print(groupedarrays)
        # print(groupedarraysKeys)
        
class Obj():
    def __init__(self, M, G:int):
        self.M = M
        self.G = G
        self.dim = M[0].shape[1]
        self.n = len(M)
        
    def initGMM(self):
        GMM = [np.random.rand(self.dim,self.dim+1) for _ in range(self.G)]
        return GMM
    
    def initPHI(self):
        PHI = [np.random.rand(self.G) for _ in range(self.n)]
        return PHI
    
    def muldiNorFunc(X, par:np.array)->float:
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
        minus = X - mu
        minus_T = minus.reshape(-1,1)
        var_inv = np.linalg.inv(var)
        value = np.exp((-1/2)*minus_T @ var_inv @ minus)/((2*np.pi)**(0.5*len(X))*np.linalg.det(var)**0.5)
        
        return value
    
    def develop(self, GMM, PHI):
        self.KSI = []
        def EStep():
            for i in range(self.n):
                ksi = np.zeros((self.M[i].shape[0], self.G))
                for j in range(self.M[i].shape[0]):
                    f_g = []
                    for g in range(self.G):
                        f_g.append(Obj.muldiNorFunc(self.M[i][j,g], GMM[g]))
                    
                    f_g = np.array(f_g)
                    ksi_j = (f_g * PHI[i])/(np.dot(f_g, PHI[i]))
                    ksi[j] = ksi_j
                self.KSI.append(ksi)
                
            for i in range(self.n):
                for g in range(self.G):
                    PHI[i][g] = np.sum(self.KSI[i][:,g])/self.M[i].shape[0]
        def MStep():
            for g in range(self.G):
                ksi_sum_g = 0
                for i in range(self.n):
                    ksi_sum_g += np.sum(self.KSI[i][:,g])
                mu = np.zeros(self.dim)
                for i in range(self.n):
                    for j in range(self.M[i].shape[0]):
                        mu += self.KSI[i][j,g]*self.M[i][j]
                mu /= ksi_sum_g
                mu = mu.reshape(-1,1)
                GMM[g][:,0] = mu
            
            for g in range(self.G):
                ksi_sum_g = 0
                for i in range(self.n):
                    ksi_sum_g += np.sum(self.KSI[i][:,g])
                cov = np.zeros((self.dim, self.dim))
                for g in range(self.G):
                    mu_g = GMM[g][:,0]
                    for i in range(self.n):
                        for j in range(self.M[i].shape[0]):
                            minus = self.M[i][j]-mu_g
                            minus_T = minus.reshape(-1,1)
                            cov += self.KSI[i][j,g]*(minus @ minus_T)
                cov /= ksi_sum_g
                GMM[g][:,1:] = cov
            print(self.KSI[0].shape)
    
        EStep()
        MStep()

    
    def EM(self, T0):
        self.GMM = self.initGMM()
        self.PHI = self.initPHI()
        for t in tqdm(range(T0), desc="EM Iterations"):
            self.develop(self.GMM, self.PHI)
        print(self.KSI[0].shape)
        return self.PHI, self.GMM
    
    def BIC(self):
        score = 0
        for i in range(self.n):
            for j in range(self.M[i].shape[0]):
                for g in range(self.G):
                    try:
                        score += self.KSI[i][j, g] * (math.log(self.PHI[i][g] + math.log(Obj.muldiNorFunc(self.M[i][j], self.GMM[g]))))
                    except IndexError:
                        print(f"IndexError at i={i}, j={j}, g={g}")
                        print(f"KSI shape: {self.KSI[i].shape}")
                        print(f"M shape: {self.M[i].shape}")
                        print(f"GMM shape: {self.GMM[g].shape}")
                        raise
        print(score)
        return score
    
    def joint(self, groupedarraysKeys):
        data = []
        for key, array in zip(groupedarraysKeys, self.PHI):
            data.append({'CompanyCode':key, 'Phis':array})
            
        self.PHI_df = pd.DataFrame(data)
        return self.PHI_df
    def save(self, path):
        self.PHI_df.to_excel(path)
        print(f"已保存到{path}下")

class Kmeans():
    def __init__(self, data):
        self.data = np.array(data)
        
    def K_means_plus(self):
        kmeans_plus = KMeans(n_clusters=10, init='k-means++') # 'k-means++' 是关键参数
        kmeans_plus.fit(self.data)
        self.labels = kmeans_plus.labels_

    def joint(self, CompanyCode):
        data_new = np.c_[CompanyCode, self.data, self.labels]
        self.df = pd.DataFrame(data_new)
        columnsCount = self.df.shape[1]
        self.df.columns = ['CompanyCode'] + list(self.df.columns[1:-1]) +['Cluster']
        return np.c_[self.data, self.labels]
    def silhouette(self):
        score = silhouette_score(self.data, self.labels)
        return score
    def save(self, path):
        self.df.to_excel(path)
        print(f"已保存到{path}下")
        

if __name__ == '__main__':
    
    dataset = DataSet('D:\\OneDrive\\3考博大业\\朱映秋老师\\论文代码复现\\Coding\\Data\\DataPlus.xlsx')
    dataset.loadData()
    dataset.DataToArr()
    obj = Obj(M = dataset.groupedarrays, G = 3)
    Phi, Gmm = obj.EM(1000)
    # obj.BIC()
    kmeans = Kmeans(Phi)
    kmeans.K_means_plus()
    PHI_new = kmeans.joint(CompanyCode=dataset.groupedarraysKeys)
    print(kmeans.silhouette())
    # kmeans.save('D:\\OneDrive\\3考博大业\\朱映秋老师\\论文代码复现\\Coding\\Data\\phi.xlsx')



    
    '''    
    以下为class DataSet和class Obj的使用示例2
    dataset = DataSet('D:\\OneDrive\\3考博大业\\朱映秋老师\\论文代码复现\\Coding\\Data\\DataPlus.xlsx')
    dataset.loadData()
    dataset.DataToArr()
    obj = Obj(M = dataset.groupedarrays, G = 4)
    Phi, Gmm = obj.EM(1000)
    df = obj.joint(dataset.groupedarraysKeys)
    obj.save('D:\\OneDrive\\3考博大业\\朱映秋老师\\论文代码复现\\Coding\\Data\\ProPlus.xlsx')
    以下为class DataSet和class Obj的使用示例2
    dataset = DataSet('D:\\OneDrive\\3考博大业\\朱映秋老师\\论文代码复现\\Coding\\Data\\Data.xlsx')
    dataset.loadData()
    dataset.DataToArr()
    obj = Obj(M = dataset.groupedarrays, G = 4)
    Phi, Gmm = obj.EM(1000)
    df = obj.joint(dataset.groupedarraysKeys)
    obj.save('D:\\OneDrive\\3考博大业\\朱映秋老师\\论文代码复现\\Coding\\Data\\Processed2.xlsx')
    '''
