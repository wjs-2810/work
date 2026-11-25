# 实验八 手写逻辑回归：期末挂科预警

Python 机器学习课程作业。从零手写逻辑回归（Sigmoid、对数损失、梯度下降），
用「每周学习时长」和「每天刷手机时长」两个特征预测学生期末是否挂科。

## 运行结果

- 数据：400 名学生，其中挂科 170 人
- 收敛：第 1138 轮（相邻两轮损失差 < 1e-6），损失 0.4161
- 参数：w0 = -0.0537，w1 = -0.6511，w2 = 0.7147
- 小明（学习 3h / 手机 7h）：挂科概率 95.2% → 预警
- 小红（学习 9h / 手机 2h）：挂科概率 1.1% → 不预警
- 分类线：x2 ≈ 0.075 + 0.911 × x1

## 目录结构

```
.
├── main.py                     # 主程序：读数据 → 手算验算 → 训练 → 画图 → 预测
├── logistic_regression.py      # 函数 1~4：sigmoid / loss / gradient / Logistic_Regression
├── draw.py                     # 画图函数：损失曲线 / 学习率对比 / 分类线
├── data/
│   └── students.csv            # 400 条学生数据
├── output/
│   ├── 图1_损失曲线.png
│   ├── 图2_不同学习率.png
│   └── 图3_分类线.png
└── run_output.txt              # 完整运行输出（已复现指导书数值）
```

## 运行方法

```bash
pip install numpy pandas matplotlib
python main.py
```

运行后会在 `output/` 重新生成三张图，并在终端打印完整结果。

## 环境

Python 3.8+，numpy，pandas，matplotlib。
