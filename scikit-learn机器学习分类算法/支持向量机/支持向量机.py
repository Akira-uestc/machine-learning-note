# 导入numpy库
import numpy as np

# 定义支持向量机类
class SVM:

    # 初始化参数
    def __init__(self, C=1.0, kernel='linear', tol=1e-3, max_iter=1000):
        self.C = C # 惩罚系数
        self.kernel = kernel # 核函数
        self.tol = tol # 容错率
        self.max_iter = max_iter # 最大迭代次数
        self.alpha = None # 拉格朗日乘子
        self.b = None # 偏置项
        self.E = None # 误差缓存
        self.K = None # 核矩阵

    # 计算核函数值
    def kernel_func(self, x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1, x2) # 线性核
        elif self.kernel == 'rbf':
            gamma = 1.0 / x1.shape[0] # 高斯核参数
            return np.exp(-gamma * np.sum((x1 - x2) ** 2)) # 高斯核
        else:
            raise ValueError('Invalid kernel') # 无效的核函数

    # 计算核矩阵
    def calc_K(self, X):
        n_samples = X.shape[0] # 样本数
        K = np.zeros((n_samples, n_samples)) # 初始化核矩阵
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel_func(X[i], X[j]) # 计算核函数值
        return K

    # 计算预测值
    def predict(self, X):
        return np.dot(self.alpha * self.y, self.K[:, X]) + self.b

    # 计算误差值
    def calc_E(self, i):
        return self.predict(i) - self.y[i]

    # 随机选择另一个变量的索引
    def select_j(self, i, n_samples):
        j = i # 初始化j为i
        while j == i:
            j = np.random.randint(0, n_samples) # 随机选择一个不等于i的索引
        return j

    # 裁剪alpha值到[L, H]区间内
    def clip_alpha(self, alpha, L, H):
        if alpha < L:
            return L
        elif alpha > H:
            return H
        else:
            return alpha

    # 训练模型
    def fit(self, X, y):
        n_samples, n_features = X.shape # 样本数和特征数
        self.alpha = np.zeros(n_samples) # 初始化alpha为零向量
        self.b = 0.0 # 初始化b为零
        self.E = np.zeros(n_samples) # 初始化误差缓存为零向量
        self.y = y # 保存标签向量
        self.K = self.calc_K(X) # 计算核矩阵

        for _ in range(self.max_iter): # 外层循环，最多迭代max_iter次
            alpha_changed = 0 # 记录alpha是否有更新

            for i in range(n_samples): # 内层循环，遍历每个样本

                E_i = self.calc_E(i) # 计算E_i

                if (y[i] * E_i < -self.tol and self.alpha[i] < self.C) or \
                    (y[i] * E_i > self.tol and self.alpha[i] > 0): 
                    # 如果alpha_i可以被优化，即违反了KKT条件

                    j = self.select_j(i, n_samples) # 随机选择另一个变量的索引j

                    E_j = self.calc_E(j) # 计算E_j

                    alpha_i_old = self.alpha[i].copy() # 保存旧的alpha_i值
                    alpha_j_old = self.alpha[j].copy() # 保存旧的alpha_j值

                    # 计算alpha_j的上下界
                    if y[i] == y[j]:
                        L = max(0, self.alpha[j] + self.alpha[i] - self.C)
                        H = min(self.C, self.alpha[j] + self.alpha[i])
                    else:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])

                    if L == H: # 如果上下界相等，跳过本次循环
                        continue

                    # 计算alpha_j的最优修改量
                    eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]

                    if eta >= 0: # 如果eta非负，跳过本次循环
                        continue

                    # 更新alpha_j值
                    self.alpha[j] -= y[j] * (E_i - E_j) / eta

                    # 裁剪alpha_j值
                    self.alpha[j] = self.clip_alpha(self.alpha[j], L, H)

                    # 更新E_j值
                    self.E[j] = self.calc_E(j)

                    if abs(self.alpha[j] - alpha_j_old) < 1e-5: # 如果alpha_j变化太小，跳过本次循环
                        continue

                    # 更新alpha_i值
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])

                    # 更新E_i值
                    self.E[i] = self.calc_E(i)

                    # 更新b值
                    b1 = self.b - E_i - y[i] * (self.alpha[i] - alpha_i_old) * \
                        self.K[i, i] - y[j] * (self.alpha[j] - alpha_j_old) * \
                        self.K[i, j]

                    b2 = self.b - E_j - y[i] * (self.alpha[i] - alpha_i_old) * \
                        self.K[i, j] - y[j] * (self.alpha[j] - alpha_j_old) * \
                        self.K[j, j]

                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0

                    alpha_changed += 1 # 增加alpha更新次数

            if alpha_changed == 0: # 如果没有alpha更新，退出循环
                break

        return self
