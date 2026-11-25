"""
画图
包含 3 个函数：draw_loss()、draw_learning_rate()、draw_decision_boundary()
"""
import numpy as np
import matplotlib.pyplot as plt

from logistic_regression import Logistic_Regression


def draw_loss(loss_list):
    """
    函数 5　画迭代轮数与损失函数的关系图
    功能：横轴为迭代轮数，纵轴为损失函数值，画出损失随迭代轮数变化的曲线
    输入：loss_list：Logistic_Regression() 返回的损失列表
    输出：无（只画图，不调用 plt.show()）
    """
    plt.figure(figsize=(8, 5))
    iterations = range(1, len(loss_list) + 1)     # 横轴：第 1 轮、第 2 轮 ……
    plt.plot(iterations, loss_list, color='royalblue', linewidth=1.5)
    plt.title('图1  迭代轮数与损失函数的关系', fontsize=14)
    plt.xlabel('迭代轮数', fontsize=12)
    plt.ylabel('损失函数值 J(w)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()


def draw_learning_rate(x, y, lr_list, num_iter):
    """
    函数 6　画不同学习率下损失函数的变化图
    功能：对 lr_list 中的每个学习率，调用 Logistic_Regression() 各训练 num_iter 轮
          （threshold 取 0，不提前停止），把各自的损失曲线画在同一张图上，用图例标出学习率
    输入：x、y：特征和标签
          lr_list：学习率列表
          num_iter：迭代轮数
    输出：无（只画图，不调用 plt.show()）
    """
    plt.figure(figsize=(8, 5))
    colors = ['seagreen', 'royalblue', 'darkorange', 'crimson', 'purple']

    for i, lr in enumerate(lr_list):
        # threshold 取 0，|本轮损失 - 上一轮损失| < 0 永远不成立，所以一定跑满 num_iter 轮
        loss_list, _ = Logistic_Regression(x, y, lr, num_iter, 0)
        plt.plot(range(1, len(loss_list) + 1), loss_list,
                 label='学习率 = {}'.format(lr),
                 color=colors[i % len(colors)], linewidth=1.2)

    plt.title('图2  不同学习率下损失函数的变化', fontsize=14)
    plt.xlabel('迭代轮数', fontsize=12)
    plt.ylabel('损失函数值 J(w)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()


def draw_decision_boundary(x, y, w):
    """
    函数 7　画分类线
    功能：画出所有同学的散点图（挂科和没挂科用不同颜色、大小区分），
          再画出分类线 w0 + w1 × x1 + w2 × x2 = 0
    输入：x：特征，形状 (m, 2)，第 1 列学习时长，第 2 列刷手机时长
          y：标签，形状 (m,)
          w：Logistic_Regression() 返回的参数 [w0, w1, w2]
    输出：无（只画图，不调用 plt.show()）
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    plt.figure(figsize=(8, 5))

    # 第一步：散点图。挂科的同学用红色 ×，没挂科的用蓝色 ○，大小也区分开
    fail = y == 1
    plt.scatter(x[~fail, 0], x[~fail, 1], marker='o', s=18,
                c='royalblue', alpha=0.7, label='没挂科（0）')
    plt.scatter(x[fail, 0], x[fail, 1], marker='x', s=30,
                c='crimson', alpha=0.8, label='挂科（1）')

    # 第二步：分类线 w0 + w1 × x1 + w2 × x2 = 0，解出 x2 = -(w0 + w1 × x1) / w2
    w0, w1, w2 = w
    x1 = np.array([x[:, 0].min(), x[:, 0].max()])
    x2 = -(w0 + w1 * x1) / w2
    plt.plot(x1, x2, color='red', linewidth=2, label='分类线  w0 + w1*x1 + w2*x2 = 0')

    plt.title('图3  学生分布与分类线', fontsize=14)
    plt.xlabel('每周课后学习时长（小时）', fontsize=12)
    plt.ylabel('每天刷手机娱乐时长（小时）', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
