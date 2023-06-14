# 导入numpy库
import numpy as np

# 定义数据点
x = np.array([随便输个数组])
y = np.array([随便输个数组])

# 计算x和y的均值
x_mean = np.mean(x)
y_mean = np.mean(y)

# 计算x和y的协方差
cov_xy = np.sum((x - x_mean) * (y - y_mean))

# 计算x的方差
var_x = np.sum((x - x_mean) ** 2)

# 根据最小二乘法公式，计算斜率和截距
slope = cov_xy / var_x
intercept = y_mean - slope * x_mean

# 打印结果
print(f"y = {slope:.2f} * x + {intercept:.2f}")
