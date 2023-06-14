# 导入所需的库
import math
import random

# 定义一个节点类，用于存储决策树的结构
class Node:
    def __init__(self, feature=None, value=None, left=None, right=None, label=None):
        self.feature = feature # 划分特征的索引
        self.value = value # 划分特征的值
        self.left = left # 左子树
        self.right = right # 右子树
        self.label = label # 叶节点的类别标签

# 定义一个函数，用于计算数据集的信息熵
def entropy(data):
    # 统计每个类别的频数
    counts = {}
    for row in data:
        label = row[-1]
        counts[label] = counts.get(label, 0) + 1
    # 计算信息熵
    ent = 0.0
    for label in counts:
        p = counts[label] / len(data)
        ent -= p * math.log(p, 2)
    return ent

# 定义一个函数，用于划分数据集
def split(data, feature, value):
    # 根据特征和值划分数据集为两部分
    left = []
    right = []
    for row in data:
        if row[feature] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

# 定义一个函数，用于选择最佳的划分特征和值
def best_split(data):
    # 初始化最佳的信息增益、特征和值
    best_gain = 0.0
    best_feature = None
    best_value = None
    # 计算数据集的信息熵
    base_entropy = entropy(data)
    # 遍历每个特征和每个可能的值
    for feature in range(len(data[0]) - 1):
        values = set([row[feature] for row in data])
        for value in values:
            # 划分数据集
            left, right = split(data, feature, value)
            # 计算信息增益
            p = len(left) / len(data)
            gain = base_entropy - p * entropy(left) - (1 - p) * entropy(right)
            # 更新最佳的信息增益、特征和值
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_value = value
    return best_gain, best_feature, best_value

# 定义一个函数，用于构建决策树
def build_tree(data):
    # 如果数据集为空，返回None
    if len(data) == 0:
        return None
    # 如果数据集中只有一种类别，返回该类别标签作为叶节点
    labels = set([row[-1] for row in data])
    if len(labels) == 1:
        return Node(label=labels.pop())
    # 选择最佳的划分特征和值
    gain, feature, value = best_split(data)
    # 如果信息增益为零，返回数据集中最多的类别标签作为叶节点
    if gain == 0:
        counts = {}
        for row in data:
            label = row[-1]
            counts[label] = counts.get(label, 0) + 1
        max_label = max(counts, key=counts.get)
        return Node(label=max_label)
    # 划分数据集为两部分，并递归构建左右子树
    left, right = split(data, feature, value)
    left_tree = build_tree(left)
    right_tree = build_tree(right)
    # 返回当前节点作为根节点
    return Node(feature=feature, value=value, left=left_tree, right=right_tree)

# 定义一个函数，用于打印决策树（辅助函数）
def print_tree(node, indent=""):
    # 如果是叶节点，打印类别标签
    if node.label is not None:
        print(indent + "Label:", node.label)
    # 如果不是叶节点，打印划分特征和值，并递归打印左右子树
    else:
        print(indent + "Feature:", node.feature, "Value:", node.value)
        print(indent + "Left:")
        print_tree(node.left, indent + "  ")
        print(indent + "Right:")
        print_tree(node.right, indent + "  ")

# 定义一个函数，用于预测新数据的类别
def predict(node, row):
    # 如果是叶节点，返回类别标签
    if node.label is not None:
        return node.label
    # 如果不是叶节点，根据划分特征和值，递归预测左右子树
    if row[node.feature] < node.value:
        return predict(node.left, row)
    else:
        return predict(node.right, row)

# 定义一个函数，用于测试决策树的准确率
def accuracy(data, tree):
    # 统计预测正确的个数
    correct = 0
    for row in data:
        label = row[-1]
        pred = predict(tree, row)
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

# 构建决策树并打印
tree = build_tree(data)
print_tree(tree)

# 测试决策树的准确率
acc = accuracy(data, tree)
print("Accuracy:", acc)
