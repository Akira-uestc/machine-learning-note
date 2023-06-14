import numpy as np

# 假设训练集
train_set = np.array([[3, 3, 1],
            [4, 3, 1],
            [1, 1, -1]])
w = np.array([0, 0]) # 权重参数
b = 0 #偏置
l = 1 # 学习率

# 更新该样本点参数
def update(item):
  global w, b # 全局变量（便于修改全局变量w和b）
  # 计算w和b ——w += l * yi * xi，b += l * yi
  w += l * item[-1] * item[:-1]
  b += l * item[-1]
  # 打印结果
  print("w={}, b={}".format(w, b))

# 检查是否有错误分类点
def check():
  # 默认无错误分类点
  flag = False
  # 检查所有样本点
  # 记录检查结果
  res = 0
  for item in train_set:
    # 计算w*xi+b
    res = (w * item[:-1]).sum() + b
    # 计算yi(w*xi+b)
    res *= item[-1]
    # 判断是否错误分类
    if res <= 0:
      # 错误分类
      flag = True
      # 更新该样本点参数
      update(item)
  return flag

if __name__ == "__main__":
  flag = False
  for i in range(100):
    # 无错误分类点，结束迭代
    if not check(): # check返回False，表示无错误分类点
      flag = True
      break
    # 有错误分类点，需继续迭代
  
  if flag:
    print("100次迭代,可以完成正确分类")
  else:
    print("100次迭代,不可完成正确分类")
