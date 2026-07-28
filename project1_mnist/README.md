# Project 1：MNIST MLP 与 CNN 对照实验

这是暑假计划里的第一个完整 PyTorch 工程。它把 W1 分散学习的 Tensor、
`nn.Module`、`DataLoader`、训练/验证、device、checkpoint 和曲线记录组成一个
可复现闭环。

完整教程见：
`1_每日任务指南/W2_工程实践02_MNIST完整教程.md`。

## 先做不联网冒烟测试

在仓库根目录执行：

```powershell
conda activate ai-learn
python -m unittest discover -s project1_mnist/tests -v
python -m project1_mnist.train --dataset synthetic --model mlp --epochs 3
```

`synthetic` 只验证数据→模型→训练→评估→保存这一整条管道，不是正式实验。

## 正式训练

第一次运行会把 MNIST 下载到仓库根目录的 `data/`：

```powershell
python -m project1_mnist.train --dataset mnist --model mlp --epochs 5
python -m project1_mnist.train --dataset mnist --model cnn --epochs 5
```

评估最佳 checkpoint：

```powershell
python -m project1_mnist.evaluate runs/mnist/mlp/best.pt
python -m project1_mnist.evaluate runs/mnist/cnn/best.pt
```

绘制训练曲线：

```powershell
python -m project1_mnist.plot_history runs/mnist/mlp/history.csv `
  --output runs/mnist/mlp/curves.png
python -m project1_mnist.plot_history runs/mnist/cnn/history.csv `
  --output runs/mnist/cnn/curves.png
```

## 预期结果

在完整 MNIST、默认参数、训练 5 epochs 的情况下，通常应看到：

- MLP 测试准确率达到约 97% 或更高；
- CNN 测试准确率达到约 99% 或接近 99%；
- `runs/mnist/{model}/` 中产生 `best.pt`、`last.pt`、`history.csv`。

数值会受硬件、PyTorch 版本和随机性影响。README 中应记录自己的真实运行结果，
不要把上述预期值写成已经完成的成果。

## 项目结构

```text
project1_mnist/
├── data.py           # MNIST / synthetic 数据与 DataLoader
├── models.py         # MLP / CNN
├── engine.py         # 训练与评估循环
├── utils.py          # 随机种子、checkpoint、CSV
├── train.py          # 训练命令行入口
├── evaluate.py       # 独立评估
├── plot_history.py   # 曲线绘制
└── tests/            # 离线冒烟测试
```

## 完成标准

- 两个模型都从命令行独立训练成功；
- 分清 train 与 test，不在测试集上更新参数；
- 保存 best/last checkpoint 和 CSV 曲线数据；
- README 填入参数量、最好准确率、曲线和 MLP/CNN 差异解释；
- 至少做一项小改动并比较，而不是只运行现成代码。
