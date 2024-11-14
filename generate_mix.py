#!/usr/bin/python2.7 
# -*- coding: utf-8 -*-

# 生成模拟数据
import random
import pandas as pd
import scipy.stats as ss
import os, shutil
import numpy as np
import matplotlib.pyplot as plt


#from sklearn.datasets.samples_generator import make_blobs 

#from zh_input import *
import config as cfg


def work(mark, N, baseCount, flag, gd):
    # 生成模拟数据

    dataSet = []

    # 生成商户交易笔数
    count = np.random.poisson(baseCount,size=N)

    tx1_count = np.random.poisson(int(baseCount * 0.2),size=N)

    tx2_count = np.random.poisson(int(baseCount * 0.7),size=N)


    # 生成每笔交易金额
    for i in range(0, N):
        temp_set = []
        name = mark + '_' + str(i)

        # 单个商户交易笔数
        count_trans = int(count[i])

        if flag == 1:
            #ad2 = ss.expon.rvs(loc=0, scale=2, size=int(count_trans))
            #ad1 = ss.norm.rvs(loc=1.5, scale=sigma, size=int(count_trans/2))
            #ad2 = ss.chi2.rvs(2, 1, size=int(count_trans))

            # loc = 2
            ad2 = ss.norm.rvs(loc=4, scale=2, size=int(count_trans))

            ad = list(ad2)
            for adi in range(len(ad)):
                ad[adi] = abs(ad[adi])


        elif flag == 2:
            ad2 = ss.expon.rvs(loc=0, scale=2, size=int(count_trans-tx1_count[i]))
            #ad2 = ss.norm.rvs(loc=2, scale=2, size=int(count_trans-tx1_count[i]))

            ad = list(ad2)

            for adi in range(len(ad)):
                ad[adi] = abs(ad[adi])

            for j in range(tx1_count[i]):
                ad.append( 10 + 2 * random.random() )

        elif flag == 3:
            ad2 = ss.expon.rvs(loc=0, scale=2, size=int( abs(count_trans-tx2_count[i]) ))
            #ad2 = ss.gamma.rvs(2, loc=0, scale=1, size=int( abs(count_trans-tx2_count[i]) ))
            #ad2 = ss.norm.rvs(loc=2, scale=2, size=int( abs(count_trans-tx2_count[i]) ))

            #for adi in range(len(ad2)):
            #    ad2[adi] = 0.1 * int(ad2[adi]/0.1)

            ad = list(ad2)

            for adi in range(len(ad)):
                ad[adi] = abs(ad[adi])

            for j in range(tx2_count[i]):
                ad.append( 4 + 2 * random.random() )

        for j in range(0, count_trans):
            # 生成单笔交易金额
            amount = int(ad[j])
            temp_set.append([name, amount, mark])

        # 打乱顺序
        random.shuffle(temp_set)
        dataSet += temp_set
        #print i
    print("-->: generating finished")

    dataSet = pd.DataFrame(dataSet, columns = [cfg.MERCHANTID, cfg.TRANSACTIONAMOUNT, cfg.mark])
    
    print("= [%d] - [%.2f] [%.2f] = %.2f ~ %.2f " % (flag, dataSet[cfg.TRANSACTIONAMOUNT].mean(),
        dataSet[cfg.TRANSACTIONAMOUNT].std(), dataSet[cfg.TRANSACTIONAMOUNT].min(),
        dataSet[cfg.TRANSACTIONAMOUNT].max()))

    dataSet.to_csv(gd+'/'+mark+'.csv',index=False)
    return dataSet[cfg.TRANSACTIONAMOUNT]

def draw():
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 20,
    }
    fig = plt.figure(figsize=(19, 6))
    for i in range(0, 3):
        print("-------> Cluster ", i+1)
        

        a = work('a_%d' % (i+1), 200, 50, i+1, 'simpic')
        #a.append(50000)
        tar = 131 + i
        ax = fig.add_subplot(tar)
        ax.grid(linestyle=':')
        ax.hist(a, bins=256)
        ax.set_title("Cluster "+str(i+1),
            fontdict = {'fontsize':22})
        plt.xlabel('Transaction amount', font2)
        plt.ylabel('Frequency', font2)
        #plt.grid(linestyle=':')

    plt.savefig('simpic'+'/sim.jpg', format='jpg')

#draw()

'''
source = 'simulation_data_mix'
if os.path.exists(source):
	shutil.rmtree(source)
os.mkdir(source)
ass = [50, 50, 50]
for i in range(1, 4):
    work('a'+str(i), ass[i-1], 50, i, 'simulation_data_mix')
'''



