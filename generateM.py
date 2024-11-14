import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal

def generate_data(mean, cov, size):
    return np.random.multivariate_normal(mean, cov, size)

def calculate(mean, cov, data):
    mvn = multivariate_normal(mean, cov)
    result = []
    for i in range(len(data)):
        result.append(mvn.pdf(data[i]))
        
    return np.array(result)
    
def plot_data(mean_cov, z):
    x = mean_cov[:,0]
    y = mean_cov[:,1]
    z = z
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')
    plt.show()

def generate_obserNum(ClusterNum:tuple, lam):
    """
    根据给定的簇数量和泊松分布参数生成观测数。
    
    Args:
        ClusterNum (tuple): 一个包含每个簇中对象数量的元组。
        lam (float): 泊松分布的均值参数。
    
    Returns:
        tuple: 一个包含每个簇中观测数的元组，观测数根据泊松分布生成。
    
    """
    return tuple([np.random.poisson(lam, num) for num in ClusterNum])
    
'''
函数目的：随机生成不同对象对应数目的正态随机化观测向量/值
接收：均值、协方差、簇_对象标签
返回：list型观测数据向量/值
'''
def generate_observation(mean, cov, cluster_label:np.ndarray):
    """
    根据给定的均值、协方差矩阵和簇标签生成观测数据列表。
    
    Args:
        mean (np.ndarray): 均值向量，维度应与协方差矩阵的维度一致。
        cov (np.ndarray): 协方差矩阵，用于描述数据的分布。
        cluster_label (np.ndarray): 簇标签数组，表示每个观测数据所属的簇，必须是一维数组！！！
    
    Returns:
        list: 包含生成的观测数据的列表，列表长度与簇标签数组长度相同。
    
    """
    # lst = []
    # for index in cluster_label:
    #     lst.append(generate_data(mean, cov, len(cluster_label)))
    lst = [generate_data(mean, cov, index) for index in cluster_label]
    return lst

'''
把我们的对象观测大数组进行合并并加标签
输入:对象观测大数组
输出：合并后的对象观测大数组并且保存到本地xlsx文件
'''
def merge_and_save(data, order=str):
    pd0 = pd.DataFrame(data[0])
    pd0['obj_num'] = order+'0'
    for index, num in enumerate(data[1:],start=1):
        pdi = pd.DataFrame(num)
        pdi['obj_num'] = order+str(index)
        pd0 = pd.concat([pd0,pdi], ignore_index=True)

    return pd0

if __name__ == '__main__':
    
    mean1 = np.array([0, 0])
    cov1 = np.array([[1, 1/2], [1/2, 1]])
    mean2 = np.array([0, 0])
    cov2 = np.array([[1, -1/2], [-1/2, 1]])
    mean = np.array([0, 0])
    cov11 = 0.2*cov1 + 0.8*cov2
    cov22 = 0.5*cov1 + 0.5*cov2
    cov33 = 0.8*cov1 + 0.2*cov2

    index = generate_obserNum((60, 100, 150), 50) 

    data1 = generate_observation(mean, cov11, index[0])
    data2 = generate_observation(mean, cov22, index[1])
    data3 = generate_observation(mean, cov33, index[2])
    pd1 = merge_and_save(data1,order='1')
    pd2 = merge_and_save(data2,order='2')
    pd3 = merge_and_save(data3,order='3')
    # pd1['cluster_num'] = 1
    # pd2 = pd.DataFrame(merge_and_save(data2))
    # pd2['cluster_num'] = 2
    # pd3 = pd.DataFrame(merge_and_save(data3))
    # pd3['cluster_num'] = 3
    print(pd1)
    print(pd2)
    print(pd3)
    pd = pd.concat([pd1, pd2, pd3], ignore_index=True)
    pd.to_excel('D:\\OneDrive\\3考博大业\\朱映秋老师\\论文代码复现\\Coding\\simulated_M.xlsx')
    # z1 = calculate(mean1, cov1, data1)
    # mean2 = np.array([0, 0])
    # cov2 = np.array([[1, -1/2], [-1/2, 1]])
    # data2 = generate_data(mean2, cov2, 10000)
    # z2 = calculate(mean2, cov2, data2)


