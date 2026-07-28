# Project 0：从零实现 micrograd

这是一个只处理标量的自动求导引擎。它不追求性能，目的只有一个：把
`loss.backward()` 背后的计算图、拓扑排序和链式法则变成自己能解释的代码。

完整学习步骤见：
`1_每日任务指南/W2_工程实践01_micrograd完整教程.md`。

## 运行

在仓库根目录执行：

```powershell
conda activate ai-learn
cd project0_micrograd
python -m unittest discover -s tests -v
python demo_train.py
```

预期：3 个测试通过；训练最终 loss 明显下降，accuracy 达到 100%。

## 文件

```text
project0_micrograd/
├── minigrad/
│   ├── engine.py        # Value 与反向传播
│   └── nn.py            # Neuron / Layer / MLP
├── tests/
│   └── test_micrograd.py
└── demo_train.py        # 4 个样本上的完整训练
```

## 你需要能回答

1. 为什么每个局部反向函数都使用 `+=`，不能用 `=`？
2. 为什么反向传播前要做拓扑排序？
3. 为什么最终 loss 必须是标量？
4. `model.zero_grad()` 和 `loss.backward()` 各自清理/生成什么？
