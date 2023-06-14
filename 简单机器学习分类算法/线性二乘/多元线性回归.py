from sklearn.linear_model import LinearRegression
import numpy as np

# 多元线性回归分析
def multiregress(*arrays) -> float:
    # 创建线性回归对象
    lr = LinearRegression()
    # 将自变量的一维数组转换为一个可以用于python线性回归计算得二维数组
    x = np.array(arrays[:-1]).T
    # 将因变量的数组转换为一个二维数组
    y = np.array(arrays[-1]).reshape(-1, 1)
    #拟合
    lr.fit(x, y)
    print('系数：', lr.coef_[0])
    print('截距：', lr.intercept_[0])
    
multiregress(*arrays)
