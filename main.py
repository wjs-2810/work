"""
实验八　手写逻辑回归：期末挂科预警
运行方法：在 logistic-lab 目录下执行 python main.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logistic_regression import sigmoid, loss, gradient, Logistic_Regression
from draw import draw_loss, draw_learning_rate, draw_decision_boundary

# 让图中能显示中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'Heiti TC', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# ---------------- 第1步 读取数据 ----------------
df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'students.csv'))
x = df[['study_hours', 'phone_hours']].values      # 特征：每周学习时长、每天刷手机时长
y = df['fail'].values                               # 标签：1 挂科，0 没挂科
print('第1步  读取数据：{} 名同学，挂科 {} 人'.format(len(y), y.sum()))

# ---------------- 第2步 手算验算：前 4 名同学，迭代 1 轮 ----------------
x4, y4 = x[:4], y[:4]
X4 = np.concatenate((np.ones((4, 1)), x4), axis=1)   # 加一列 1
h0 = sigmoid(np.dot(X4, np.zeros(3)))                 # 参数全为 0 时的 h
loss_list, w = Logistic_Regression(x4, y4, 0.1, 1, 1e-6)
print('\n第2步  手算验算（前 4 名同学，w 初始为 0，学习率 0.1，迭代 1 轮）')
print('       初始 h    =', h0)
print('       初始损失  = {:.4f}'.format(loss(h0, y4)))
print('       梯度      =', np.round(gradient(X4, h0, y4), 4))
print('       更新后 w  =', np.round(w, 4))
print('       更新后损失 = {:.4f}'.format(loss_list[0]))

# ---------------- 第3步 梯度下降训练 ----------------
lr = 0.01            # 学习率
num_iter = 10000     # 最多迭代轮数
threshold = 1e-6     # 收敛阈值：相邻两轮损失之差小于它就停止
loss_list, w = Logistic_Regression(x, y, lr, num_iter, threshold)

print('\n第3步  梯度下降训练（学习率 {}，收敛阈值 {}）'.format(lr, threshold))
for i in range(0, len(loss_list), 100):
    print('       第 {:>4} 轮  损失 = {:.6f}'.format(i + 1, loss_list[i]))
print('       第 {:>4} 轮  损失 = {:.6f}'.format(len(loss_list), loss_list[-1]))
if len(loss_list) < num_iter:
    print('       → 第 {} 轮收敛，停止迭代'.format(len(loss_list)))
else:
    print('       → 迭代满 {} 轮仍未收敛'.format(num_iter))
print('       → 参数 w0 = {:.4f}，w1 = {:.4f}，w2 = {:.4f}'.format(w[0], w[1], w[2]))

# ---------------- 第4步 画图 ----------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
draw_loss(loss_list)
plt.savefig(os.path.join(OUTPUT_DIR, '图1_损失曲线.png'))
draw_learning_rate(x, y, [0.001, 0.01, 0.1], 3000)
plt.savefig(os.path.join(OUTPUT_DIR, '图2_不同学习率.png'))
draw_decision_boundary(x, y, w)
plt.savefig(os.path.join(OUTPUT_DIR, '图3_分类线.png'))
print('\n第4步  画图：3 张图已保存到 output 文件夹')

# ---------------- 第5步 预测 ----------------
print('\n第5步  预测')
for name, study_hours, phone_hours in [('小明', 3, 7), ('小红', 9, 2)]:
    probability = sigmoid(np.dot([1, study_hours, phone_hours], w))
    print('       {}：每周学习 {} 小时，每天刷手机 {} 小时 → 挂科概率 {:.1%}，{}'.format(
        name, study_hours, phone_hours, probability, '预警' if probability >= 0.5 else '不预警'))

plt.show()
