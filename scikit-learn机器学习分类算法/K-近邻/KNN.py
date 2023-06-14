# 导入所需的库
import math
import random

# 定义一个函数，用于计算两个数据点之间的欧氏距离
def distance(a, b):
    # 假设a和b都是列表，且长度相同
    # 除去最后一个元素（类别标签）
    a = a[:-1]
    b = b[:-1]
    # 计算欧氏距离
    dist = 0.0
    for i in range(len(a)):
        dist += (a[i] - b[i]) ** 2
    dist = math.sqrt(dist)
    return dist

# 定义一个函数，用于找到数据集中距离给定数据点最近的K个邻居
def find_neighbors(data, row, k):
    # 计算每个数据点与给定数据点的距离，并存储为一个列表
    distances = []
    for point in data:
        dist = distance(point, row)
        distances.append((point, dist))
    # 对距离列表按照距离从小到大进行排序
    distances.sort(key=lambda x: x[1])
    # 取出前K个邻居，并返回它们的类别标签
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0][-1])
    return neighbors

# 定义一个函数，用于根据K个邻居的类别标签，预测给定数据点的类别
def predict(data, row, k):
    # 找到K个邻居
    neighbors = find_neighbors(data, row, k)
    # 统计每个类别的频数
    counts = {}
    for label in neighbors:
        counts[label] = counts.get(label, 0) + 1
    # 返回频数最高的类别作为预测结果
    max_label = max(counts, key=counts.get)
    return max_label

# 定义一个函数，用于测试K-近邻算法的准确率
def accuracy(data, k):
    # 统计预测正确的个数
    correct = 0
    for row in data:
        label = row[-1]
        pred = predict(data, row, k)
        if label == pred:
            correct += 1
    # 计算准确率
    acc = correct / len(data)
    return acc

# 生成一些随机数据作为示例（特征为0-9，类别为A或B）
random.seed(42)
data = []
for i in range(100):
    row = [random.randint(0, 9) for _ in range(4)]
    label = "A" if sum(row) < 20 else "B"
    row.append(label)
    data.append(row)

# 测试不同的K值对准确率的影响
for k in range(1, 11):
    acc = accuracy(data, k)
    print("K:", k, "Accuracy:", acc)
