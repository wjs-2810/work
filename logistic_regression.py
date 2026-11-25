"""
逻辑回归（参照 PPT【例8.4】）
包含 4 个函数：sigmoid()、loss()、gradient()、Logistic_Regression()
"""
import numpy as np


def sigmoid(z):
    """
    函数 1　定义 Sigmoid 函数
    功能：计算 1 / (1 + e^(-z))
    输入：z：一个数，或 numpy 数组
    输出：与 z 形状相同的结果，取值在 0～1 之间
    """
    z = np.asarray(z, dtype=float)
    # 把 z 限制在不会让 exp() 溢出的范围内（正常数据用不到，只是防止极端值报警）
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def loss(h, y):
    """
    函数 2　定义对数损失函数
    功能：计算所有样本对数损失的平均值：平均值( -y × log(h) - (1 - y) × log(1 - h) )
    输入：h：sigmoid 函数的结果（每名同学挂科的概率），numpy 数组，形状 (m,)
          y：真实标签，numpy 数组，形状 (m,)，1 表示挂科，0 表示没挂科
    输出：损失值（一个数）
    """
    h = np.asarray(h, dtype=float)
    y = np.asarray(y, dtype=float)
    m = len(y)                     # 样本个数
    # 概率可能正好等于 0 或 1，log(0) 会出错，先夹到 1e-15 与 1-1e-15 之间
    h = np.clip(h, 1e-15, 1 - 1e-15)
    cost = -y * np.log(h) - (1 - y) * np.log(1 - h)   # 每个同学的对数损失
    return np.sum(cost) / m                            # 所有同学损失的平均值


def gradient(X, h, y):
    """
    函数 3　定义梯度计算函数
    功能：用矩阵计算梯度：X 的转置 × (h - y) ÷ m，m 是样本个数
    输入：X：第一列全为 1 的特征矩阵，形状 (m, 3)
          h：sigmoid 函数的结果，形状 (m,)
          y：真实标签，形状 (m,)
    输出：梯度，numpy 数组，形状 (3,)，与参数 w 一一对应
    """
    h = np.asarray(h, dtype=float)
    y = np.asarray(y, dtype=float)
    m = len(y)                       # 样本个数
    return np.dot(X.T, (h - y)) / m  # X 的转置 × (h - y) ÷ m


def Logistic_Regression(x, y, lr, num_iter, threshold):
    """
    函数 4　逻辑回归过程（梯度下降）
    功能：
      1. 在 x 前面加一列 1，得到 X；参数 w 全部初始化为 0
      2. 梯度下降迭代，每一轮：
           z = X 与 w 的矩阵乘积 → h = sigmoid(z) → 计算梯度 → w = w - lr × 梯度
           → 用更新后的 w 重新算 h，计算损失，存入 loss_list
      3. 收敛判断：从第 2 轮起，本轮损失与上一轮损失之差的绝对值 < threshold 时，停止迭代；
         迭代满 num_iter 轮也停止
    输入：x：特征，numpy 数组，形状 (m, 2)（不含 1 那一列）
          y：标签，numpy 数组，形状 (m,)
          lr：学习率
          num_iter：最多迭代轮数
          threshold：收敛阈值
    输出：loss_list：列表，loss_list[i] 是第 i+1 轮的损失；迭代了多少轮就是 len(loss_list)
          w：训练好的参数 [w0, w1, w2]，w0 是截距
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = len(y)                                             # 样本个数

    # ① 初始化：x 前面加一列 1 得到 X，参数 w 全部设为 0
    X = np.concatenate((np.ones((m, 1)), x), axis=1)
    w = np.zeros(X.shape[1])                               # 形状 (3,)，即 [w0, w1, w2]

    loss_list = []       # 每轮的损失
    last_loss = None     # 上一轮的损失，用来判断收敛

    for i in range(num_iter):
        # ② 计算 h
        z = np.dot(X, w)
        h = sigmoid(z)
        # ③ 计算梯度
        g = gradient(X, h, y)
        # ④ 更新参数
        w = w - lr * g
        # ⑤ 用更新后的 w 重新算 h，计算本轮损失
        h = sigmoid(np.dot(X, w))
        cur_loss = loss(h, y)
        loss_list.append(cur_loss)
        # ⑥ 判断收敛：从第 2 轮起，本轮损失与上一轮损失之差的绝对值 < threshold 就停止
        if last_loss is not None and abs(cur_loss - last_loss) < threshold:
            break
        last_loss = cur_loss

    return loss_list, w
