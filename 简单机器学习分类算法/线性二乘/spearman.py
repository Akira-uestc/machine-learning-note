# 定义两个变量的值
x = [输入数组]
y = [输入数组]

# 计算样本大小
n = len(x)

# 两个变量的秩次
x_rank = sorted(range(n), key=lambda i: x[i])
y_rank = sorted(range(n), key=lambda i: y[i])

# 计算秩次之差，平方求和
d_square_sum = sum((x_rank[i] - y_rank[i]) ** 2 for i in range(n))

rho = 1 - (6 * d_square_sum) / (n * (n ** 2 - 1))

print(f"rho = {rho:.2f}")
