import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
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
    def __init__(self, M, G:int):
        self.M = M
        self.G = G
        self.dim = M[0].shape[1]
        self.n = len(M)
        
    def initGMM(self):
        all_data = np.concatenate(self.M, axis=0)
        gmm = GaussianMixture(
        n_components=self.G, 
        random_state=0, 
        max_iter=0, 
        reg_covar=1e-6,  # 增加正则化项
        init_params='k-means++' )
        gmm.fit(all_data)
        initial_means = gmm.means_
        initial_covariances = gmm.covariances_
        combined_arrays = []
        for i in range(self.G):
            mean = initial_means[i]
            covariance = initial_covariances[i]
            combined_array = np.hstack((mean.reshape(-1, 1), covariance))
            combined_arrays.append(combined_array)
            
        return combined_arrays
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
        minus = X - mu

        minus_T = minus.reshape(-1,1)
        var_inv = np.linalg.inv(var)
        value = np.exp((-1/2)*minus_T.T @ var_inv @ minus_T)/((2*np.pi)**(0.5*self.dim)*np.linalg.det(var)**0.5)
        value = float(value)

        return value
    
    def develop(self, GMM, PHI):
        self.KSI = []
        def EStep():
            for i in range(self.n):
                ksi = np.zeros((self.M[i].shape[0], self.G))
                for j in range(self.M[i].shape[0]):
                    f_g = np.array([self.muldiNorFunc(self.M[i][j], GMM[g]) for g in range(self.G)])
                    # f_g = []
                    # for g in range(self.G):
                    #     f_g.append(self.muldiNorFunc(self.M[i][j], GMM[g]))
                    # f_g = np.array(f_g)
                    denominator = np.dot(f_g, PHI[i])
                    ksi_j = (f_g * PHI[i])/denominator
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

                GMM[g][:,0] = mu.ravel()
            
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
        EStep()
        MStep()

    def EM(self, T0):
        self.GMM = self.initGMM()
        self.PHI = self.initPHI()
        for t in tqdm(range(T0), desc="EM Iterations"):
            self.develop(self.GMM, self.PHI)
        return self.PHI, self.GMM
    
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
    dataset.loadData(sheetName="Sheet2")
    dataset.DataToArr()
    obj = Obj(M = dataset.groupedarrays, G = 3)
    obj.initGMM()
    obj.initPHI()
    Phi, Gmm = obj.EM(100)
    kmeans = Kmeans(Phi)
    kmeans.K_means_plus()
    PHI_new = kmeans.joint(CompanyCode=dataset.groupedarraysKeys)
    print(kmeans.silhouette())


