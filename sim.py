import numpy as np 
import random
import pandas as pd 
import math

def sim(n, K, B, sigma, n_block):
	# 产生模拟的网络模型
	# n: 节点总个数
	# K: 类别数
	# B: 矩阵块
	# sigma: 噪音
	# n_block: 分成块数

	# 先决定哪些节点是一类的
	# label-全部随机
	pi = np.ones(K) / K
	labels = np.random.choice(range(K), size=n, replace=True, p=pi)


	# 个体效应 N 维
	q_sigma = sigma = 0.2
	Qn = np.random.uniform(-0.5,0.5,n)
	for i in range(n):
		Qn[i] = math.exp(-Qn[i])


	# 产生全局的相似度矩阵
	S = np.zeros([n,n])
	for i in range(n-1):
		# 噪声
		noise = np.random.randn(n)*sigma + 1
		
		for j in range(i+1,n):
			# 参考相似度（簇之间的）
			_p = B[ labels[i] ][ labels[j] ]
			_r = _p * noise[j-i] * Qn[i] * Qn[j] 
			_r = min(1, max(_r, 0)) # 确保在[0,1]内
			S[i][j] = S[j][i] = _r
    
    
	# 分块
	# 所有ID的列表
	nlist = pd.Series(range(n))

	blocks = {}
	block_ids = {}
	block_size = int(n / n_block)
	for i in range(n_block):
		_left = max(i*block_size, 0)
		_right = min( (i+1)*block_size, n)

		# 随机链接
		#_ss = int(0.5*n/n_block)
		#_sample_list = pd.Series( list(set(range(n)) - set(range(_left,_right)) ) )
		#_add = _sample_list.sample(_ss, replace=False) # 150 贼好
	
		_need_list = list(range(_left, _right)) #+ list(_add)

		blocks[i] = S[:, _need_list] # 每个worker仅取部分列
	
		block_ids[i] = _need_list


	return S, blocks, block_ids, labels
			