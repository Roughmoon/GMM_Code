import pandas as pd
import numpy as np
data = pd.read_excel("D:\\OneDrive\\3考博大业\朱映秋老师\\论文代码复现\\Coding\\Data\\ProPlus.xlsx")
def parse_floats(s):
    # 去除字符串两端的方括号，然后使用空格分割字符串，并将每个部分转换为浮点数
    return list(map(float, s[1:-1].split()))

# 应用这个函数到 'Phis' 列的每一个元素，并将结果转换为NumPy数组
data['Phis'] = data['Phis'].apply(parse_floats)

# 由于每个元素现在都是一个列表，我们可以使用列表推导式结合NumPy来创建一个二维数组
phis_2d_array = np.array([np.array(x) for x in data['Phis']])

print(phis_2d_array)