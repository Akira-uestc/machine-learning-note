import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib as mpl  # 绘图地图包
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import datasets
from sklearn.model_selection import train_test_split
 
 
def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]
 
# 数据预处理
if __name__ == "__main__":
    path = u'iris.data'
    # sklearn的数据集， 鸢尾花
    iris = datasets.load_iris()
    x = iris.data[:, :2]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=666)
    # 用pipline建立模型
    # StandardScaler()作用：去均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本。 StandardScaler对每列分别标准化。
    # LogisticRegression（）建立逻辑回归模型
    lr = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(multi_class="multinomial", solver="newton-cg"))])
    lr.fit(X_train, y_train)
 
    # 画图准备
    N, M = 500, 500
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)
    x_test = np.stack((x1.flat, x2.flat), axis=1)
 
    # 开始画图
    cm_light = mpl.colors.ListedColormap(['#00FF00', '#C0C000', '#00BFFF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'yellow', 'b'])
    y_hat = lr.predict(x_test)  # 预测值
    y_hat = y_hat.reshape(x1.shape)  # 使之与输入的形状相同
    plt.pcolormesh(x1, x2, y_hat, shading='auto', cmap=cm_light)  # 预测值的显示 其实就是背景
    plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), edgecolors='k', s=100, cmap=cm_dark)  # 样本的显示
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid()
    plt.savefig('2.png')
    plt.show()
    # 训练集上的预测结果
    y_hat = lr.predict(x)  # 回归的y
    y = y.ravel()  # 变一维，ravel将多维数组降位一维
    print(y)
    result = y_hat == y  # 回归的y和真实值y比较
    print(y_hat)
    print(result)
    acc = np.mean(result)  # 求平均数
    print('准确率: %.2f%%' % (100 * acc))
