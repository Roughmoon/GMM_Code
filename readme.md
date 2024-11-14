# 1. 代码说明

> 代码分为两部分，一部分是三个类，一部分为创建对象及执行方法

## 类1：class DataSet()

### 1)代码内容

```python
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
        self.groupedarrays = groupedarrays#关键属性1
        self.groupedarraysKeys = groupedarraysKeys#关键属性2
```

### 2)调用方式(在if __name__ == '__main__': 下)

```python
    dataset = DataSet('D:\\PythonClass\\PythonProjects\\GMMCoding\\DataPlus.xlsx')#创建读取文件对象
    dataset.loadData(sheetName="Sheet7")#读取文件的table
    dataset.DataToArr()#处理table，并对dataset对象赋予两个关键属性，一个关键属性是self.groupedarrays，是一个列表，是要接下来要调用的数据。一个关键属性是self.groupedarraysKeys，是公司名。
```



### 3)解释说明

本类就一个作用：读取观测对象数据文件并对文件进行处理。

关键的是self.groupedarrays，对如以下的图表进行了如下操作：

读取数据→删除缺失值的行→标准化除第一列以外的列值→根据CompanyCode的值进行分组→将这些组按顺序放进列表→得到self.groupedarrays

| CompanyCode | Dim1         | Dim2         |
| ----------- | ------------ | ------------ |
| 1_0         | -0.103966614 | -0.569790128 |
| 1_0         | 0.850443927  | 0.133408255  |
| 1_0         | 0.154469205  | -0.060352029 |
| 1_0         | 0.408194812  | 0.570141039  |

## 类2：class Obj()

### 1)代码内容

```python
class Obj():
    def __init__(self, M, G: int, i=0, sum=10):
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
        mu = np.random.randn(self.dim) * 10
        # 生成协方差矩阵
        A = np.random.randn(self.dim, self.dim) * 10
        cov = A @ A.T  # 确保协方差矩阵是对称正定的
        return mu, cov

    def initGMM(self):
        GMM = []
        for _ in range(self.G):
            mu, cov = self.generate_random_multivariate_normal()
            GMM.append(np.hstack((mu[:, np.newaxis], cov)))
        return GMM
    
    def initPHI(self):
        PHI = [np.full(self.G, 1.0 / self.G) for _ in range(self.n)]
        return PHI

    def muldiNorFunc(self, X, par: np.array) -> float:
        """
        计算多元高斯分布的概率密度函数值
        参数：
            dim: 维度
            X: 样本点
            par: 参数，包含均值和协方差
        返回值：
            概率密度函数值
        """
        mu = par[:, 0]
        var = par[:, 1:]
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
        
        # 添加一个小的正数对角矩阵以确保正定性
        var += np.eye(var.shape[0]) * 1e-6

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
                    denominator = np.dot(f_g, PHI[i])
                    if denominator == 0:
                        raise ValueError("分母为零，可能导致数值不稳定")
                    ksi_j = (f_g * PHI[i]) / denominator
                    ksi[j] = ksi_j
                self.KSI.append(ksi)
                
            for i in range(self.n):
                for g in range(self.G):
                    PHI[i][g] = np.sum(self.KSI[i][:, g]) / self.M[i].shape[0]
        
        def MStep():
            for g in range(self.G):
                mu = np.zeros(self.dim)
                ksi_sum_g = 0
                for i in range(self.n):
                    ksi_sum_g += float(np.sum(self.KSI[i][:, g]))
                for i in range(self.n):
                    for j in range(self.M[i].shape[0]):
                        mu += self.KSI[i][j, g] * self.M[i][j]
                mu /= ksi_sum_g
                GMM[g][:, 0] = mu
            
            for g in range(self.G):
                cov = np.zeros((self.dim, self.dim))
                ksi_sum_g = 0
                for i in range(self.n):
                    ksi_sum_g += float(np.sum(self.KSI[i][:, g]))
                for i in range(self.n):
                    for j in range(self.M[i].shape[0]):
                        Mij_minus_mu = self.M[i][j] - GMM[g][:, 0]
                        Mij_minus_mu = Mij_minus_mu.reshape(-1, 1)
                        cov += self.KSI[i][j, g] * (Mij_minus_mu @ Mij_minus_mu.T)
                cov /= ksi_sum_g
                
                # 添加一个小的正数对角矩阵以确保正定性
                cov += np.eye(cov.shape[0]) * 1e-6
                GMM[g][:, 1:] = cov
        EStep()
        MStep()

    def EM(self, T0):
        self.GMM = self.initGMM()
        self.PHI = self.initPHI()
        np.set_printoptions(precision=3)
        for t in tqdm(range(T0), desc="EM Iterations"):
            self.develop(self.GMM, self.PHI)
        return self.PHI, self.GMM
```

### 2)调用方式

```python
    obj = Obj(M = dataset.groupedarrays, G = 2)
    Phi, Gmm = obj.EM(100)
    print('Phi:\n', Phi)
    print('Gmm:\n', Gmm)
```

### 3)解释说明

主要进行数据说明，在该类中，有四种关键数据详细说明：

#### [1] M

> 在该类中M的体现形式为：self.M = M，是导入的数据，代表各个对象及其观测值

M为一个列表装载了n个数组，n个数组中每一个又装有不同数量（对象的观测数目）的一维数组，形式如下：
$$ 
(\begin{bmatrix}... 
  &... \\...
  &... \\...
  & ...\\...
  &...
\end{bmatrix}_{v_{1}×dim}
,
\begin{bmatrix}... 
  &... \\...
  &... \\...
  & ...\\...
  &...
\end{bmatrix}_{v_{2}×dim}
...
,\begin{bmatrix}... 
  &... \\...
  &... \\...
  & ...\\...
  &...
\end{bmatrix}_{v_{n}×dim})
$$ 
其中小括号()代表列表，内部[]为各个二维数组

#### [2] GMM

> 在该类中GMM的体现形式为：self.GMM，G为GMM中高斯分布的数目，self.GMM是高斯列，其中存储一定数量的高斯函数的均值和协方差

GMM为一个列表装载了G个高斯分布属性，这里的属性是指第一列为均值，第二列及以后为协方差矩阵，形式如下：
$$ 
(\begin{bmatrix}\mu_{11} 
  \\\mu_{12} 
  
  & \begin{bmatrix} cov\end{bmatrix}_{1}  \\\mu_{1dim} 
  
\end{bmatrix}_{1}
,
\begin{bmatrix}\mu_{21} 
 \\\mu_{21} 
  
  & \begin{bmatrix} cov\end{bmatrix}_{2}\\\mu_{2dim} 

\end{bmatrix}_{2}
...
,
\begin{bmatrix}\mu_{G1} 
 \\\mu_{G1} 
  
  & \begin{bmatrix} cov\end{bmatrix}_{G}\\\mu_{Gdim} 

\end{bmatrix}_{G})
$$ 


#### [3] PHI

> 在该类中PHI的体现形式为：self.PHI，是每个对象的GMM各高斯元素的占比

PHI为一个列表装载了n个数组，每个数组为G行1列，形式如下：
$$
(

\begin{bmatrix}
 .\\.
 \\
 .\\G维

\end{bmatrix}_{1}
,\begin{bmatrix}
 .\\.
 \\
 .\\G维

\end{bmatrix}_{2}...\begin{bmatrix}
 .\\.
 \\
 .\\G维

\end{bmatrix}_{n}


)
$$

#### [4] KSI

> 在该类中KSI的体现形式维：self.KSI，是计算中间值，对应《统计研究》中的$\psi_{ijk}$

KSI为一个列表，有n个数组，对应n个对象，用$i$索引，每个数组的维度为$v_{i}×G$，形式如下：
$$
(
\begin{bmatrix}
\psi_{111}&\psi_{112} &...&\psi_{11G}
\\\psi_{121}&\psi_{122} &...&\psi_{12G}
\\...&...&...&...&
\\\psi_{1v_{1}1}&\psi_{1v_{1}2} &...&\psi_{1v_{1}G}
\end{bmatrix}

\begin{bmatrix}
\psi_{211}&\psi_{212} &...&\psi_{21G}
\\\psi_{221}&\psi_{222} &...&\psi_{22G}
\\...&...&...&...&
\\\psi_{2v_{2}1}&\psi_{2v_{2}2} &...&\psi_{2v_{2}G}
\end{bmatrix}
 
...

\begin{bmatrix}
\psi_{n11}&\psi_{n12} &...&\psi_{n1G}
\\\psi_{n21}&\psi_{n22} &...&\psi_{n2G}
\\...&...&...&...&
\\\psi_{nv_{n}1}&\psi_{nv_{n}2} &...&\psi_{nv_{n}G}
\end{bmatrix}
)
$$

### 4）其余说明

在代码debug过程中总是出现过矩阵无法计算的情况，丢给通义千问后给我添加了“添加小的正数矩阵”的代码，出现在66行和115行处。

## 类3：class Kmeans()

### 1）代码内容

```python
class Kmeans():
    def __init__(self, data):
        self.data = np.array(data)

    def K_means_plus(self, clusters):
        kmeans_plus = KMeans(n_clusters=clusters, init='k-means++') # 'k-means++' 是关键参数
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
```

### 2）调用方式

```python
    kmeans = Kmeans(Phi)
    kmeans.K_means_plus(clusters = 3)
    PHI_new = kmeans.joint(CompanyCode=dataset.groupedarraysKeys)
    kmeans.save('D:\\PythonClass\\PythonProjects\\GMMCoding\\Simulated_Result_100.xlsx')
```

### 3）解释说明

在对数据进行“分布因子模型”求解之后，获得每个对象的PHI值，根据这些PHI值进行K-means聚类，并将聚类结果保存。



# 2. 数学运算逻辑

> 算法的数学运算集中在class Obj中的develop()方法中

## 1）代码内容

```python
def develop(self, GMM, PHI):
        self.KSI = []#设空KSI列表
        def EStep():
            for i in range(self.n):
                ksi = np.zeros((self.M[i].shape[0], self.G))
                for j in range(self.M[i].shape[0]):
                    f_g = np.array([self.muldiNorFunc(self.M[i][j], GMM[g]) for g in range(self.G)])
                    denominator = np.dot(f_g, PHI[i])
                    if denominator == 0:
                        raise ValueError("分母为零，可能导致数值不稳定")
                    ksi_j = (f_g * PHI[i]) / denominator
                    ksi[j] = ksi_j
                self.KSI.append(ksi)
                
            for i in range(self.n):
                for g in range(self.G):
                    PHI[i][g] = np.sum(self.KSI[i][:, g]) / self.M[i].shape[0]
        
        def MStep():
            for g in range(self.G):
                mu = np.zeros(self.dim)
                ksi_sum_g = 0
                for i in range(self.n):
                    ksi_sum_g += float(np.sum(self.KSI[i][:, g]))
                for i in range(self.n):
                    for j in range(self.M[i].shape[0]):
                        mu += self.KSI[i][j, g] * self.M[i][j]
                mu /= ksi_sum_g
                GMM[g][:, 0] = mu
            
            for g in range(self.G):
                cov = np.zeros((self.dim, self.dim))
                ksi_sum_g = 0
                for i in range(self.n):
                    ksi_sum_g += float(np.sum(self.KSI[i][:, g]))
                for i in range(self.n):
                    for j in range(self.M[i].shape[0]):
                        Mij_minus_mu = self.M[i][j] - GMM[g][:, 0]
                        Mij_minus_mu = Mij_minus_mu.reshape(-1, 1)
                        cov += self.KSI[i][j, g] * (Mij_minus_mu @ Mij_minus_mu.T)
                cov /= ksi_sum_g
                
                # 添加一个小的正数对角矩阵以确保正定性
                cov += np.eye(cov.shape[0]) * 1e-6
                GMM[g][:, 1:] = cov
        EStep()
        MStep()
```

### 2）解释说明

#### EStep：对应《统计研究》

![image 20241022091646987](https://imgur.la/images/2024/10/22/image-20241022091646987.png)

#### MStep：对应《统计研究》

![image 20241022091723095](https://imgur.la/images/2024/10/22/image-20241022091723095.png)

其中PHI值放在EStep中进行更新

