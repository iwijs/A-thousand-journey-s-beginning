# W3 工程实践与理论讲义：CIFAR-10 CNN 训练工程逐行精读

> 项目位置：`2_成果作品/A-thousand-journey-s-beginning/project2_cifar10`
> 前置工程：`project1_mnist` 与 `W2_工程实践02_MNIST完整教程.md`
> 基础代码快照：2026-08-10；2026-08-16 已补三档增强、完整断点恢复和真实 baseline。
> 目标：先看懂“为什么要这样设计实验”，再从命令入口一路解释数据划分、增强、CNN、
> 梯度、验证集选模、日志、checkpoint、曲线和离线测试，最终能独立修改和验证工程。

## 0. 怎样使用这份讲义

这不是一篇只需要从头读到尾的文章。建议在 VS Code 中左右分屏：

- 左侧打开 `project2_cifar10` 中的真实源码；
- 右侧打开本讲义；
- 第一次先读第 1–4 节，建立工程地图，不要立即陷入某个 Python 语法细节；
- 第二次按 `data.py → models.py → engine.py → utils.py → train.py` 的顺序精读；
- 第三次再读 `evaluate.py → plot_history.py → tests/test_smoke.py`；
- 每读完一个函数，合上讲义，用自己的话回答“输入、输出、副作用、不变量”；
- 每读完一个文件，运行本节给出的最小验证，不要把“看起来熟悉”误当成“真正理解”。

讲义中的 `L16` 表示当前文件第 16 行，`L19–26` 表示一个跨多行的完整表达式。空行只
负责视觉分组；右括号、列表结束符等纯结构行会与它闭合的表达式一起解释。除此之外，
每一条有语义的源码都在行号表中覆盖。以后源码变化导致行号移动时，以函数名和实际代码
内容定位。

不需要逐行阅读的生成物：

- `__pycache__/`、`*.pyc`：Python 自动生成的字节码缓存；
- `data/`：下载后的数据集；
- `runs/`：训练产生的 JSON、CSV、checkpoint 和 PNG；
- `*.pt.tmp`：checkpoint 原子替换前的临时文件，正常完成后不会保留。

学习这份工程时，要同时追踪四条“流”：

1. **控制流**：哪个文件、哪个函数先执行；
2. **数据流**：Tensor 的内容、shape、dtype、device 怎样变化；
3. **梯度流**：loss 怎样沿计算图影响参数；
4. **状态流**：模型参数、BatchNorm 统计量、AdamW 状态、scheduler、随机种子、CSV 和
   checkpoint 怎样变化。

## 1. W3 到底比 W2 增加了什么

W2 的核心问题是：“能否用 PyTorch 跑通一个最小分类闭环？”
W3 的核心问题变成：“能否构造一个实验协议正确、过程可追踪、结果可比较的训练工程？”

| 方面 | W2 MNIST | W3 CIFAR-10 | 新知识 |
|---|---|---|---|
| 输入 | 灰度 `(1,28,28)` | RGB `(3,32,32)` | 多通道图像 |
| 数据协议 | 官方 train/test | 官方 train 再分 train/validation，test 隔离 | 数据泄漏与模型选择 |
| 增强 | 无随机增强 | crop + flip 只用于 train | 不变性假设 |
| 模型 | MLP/两层 CNN | 六层卷积基础 CNN | 更深层级表征、全局平均池化 |
| 优化 | Adam，固定学习率 | AdamW + weight decay + cosine | 正则化与学习率调度 |
| 日志 | train/test 指标 | train/val、学习率、耗时 | 实验诊断 |
| 配置 | checkpoint 内保存 | 额外保存 `config.json` | 可追溯性 |
| checkpoint | test accuracy 选 best | validation accuracy 选 best | 严格选模协议 |
| 测试 | shape、Dataset、单轮更新 | 划分、增强、mock、CLI、best/last 语义 | 测试实验契约 |

最关键的升级不是 CNN 多了几层，而是 **test 不再参与训练期决策**。

## 2. 项目规划、目的与完成边界

### 2.1 这个项目要解决什么

项目有三个层次的目标：

1. **算法层**：用基础 CNN 对 CIFAR-10 十分类；
2. **工程层**：构造可配置、可记录、可保存、可独立评估的训练闭环；
3. **实验方法层**：只用 validation 选择模型，冻结方案后再看 test。

### 2.2 当前已经完成什么

- 基础 CNN 与真实 CIFAR-10 数据接口；
- 可复现的 train/validation 索引划分；
- train-only 随机增强；
- `none/basic/strong` 三档可配置增强；
- AdamW 与 cosine scheduler；
- `history.csv`、`config.json`、`best.pt`、`last.pt`、`curves.png`；
- 无需联网的 6 个 smoke/integration tests；
- synthetic 端到端训练、独立评估和曲线验证。
- model/optimizer/scheduler/RNG/DataLoader generator 的 epoch 边界恢复；
- 30 轮真实 CIFAR-10 GPU baseline 与一次独立官方 test 评估。

### 2.3 当前没有完成什么

- 没有 AMP 混合精度、TensorBoard/W&B、多 seed、early stopping；
- 基础 CNN 只有一个 seed，不提供方差估计；
- ResNet 与 residual 消融位于后续 `project3_resnet`。

因此可以说“W3 工程管道与单 seed baseline 已完成”，不能说“获得稳定的多 seed 结论”或
“达到 CIFAR-10 SOTA”。

### 2.4 每个文件先做什么

在逐行阅读前，先记住职责地图：

| 文件 | 核心职责 | 不应该负责 |
|---|---|---|
| `__init__.py` | 定义包的公开 API | 训练或数据下载 |
| `data.py` | transform、划分、Dataset、DataLoader | 模型和反向传播 |
| `models.py` | BasicCNN 与模型工厂 | 数据加载、日志 |
| `engine.py` | 单轮训练、统一评估、device 解析 | argparse、输出目录 |
| `utils.py` | seed、JSON、CSV、checkpoint | 决定何时保存 best |
| `train.py` | 参数解析、对象装配、epoch 主循环 | 重复实现 Dataset/CNN |
| `evaluate.py` | 独立加载 best，在 held-out test 评估 | 更新参数或选择 best |
| `plot_history.py` | CSV → PNG | 重新训练模型 |
| `tests/test_smoke.py` | 验证接口和实验协议 | 证明真实数据最终精度 |
| `requirements.txt` | 声明运行依赖 | 锁定完整环境状态 |
| `README.md` | 给使用者提供命令、事实与边界 | 替代真实源码和测试 |

## 3. 一条训练命令究竟发生了什么

从主仓根目录执行：

```powershell
conda activate ai-learn
python -m project2_cifar10.train --dataset cifar10 --epochs 30
```

控制流：

```text
Python 以模块方式加载 project2_cifar10.train
    ↓
train.py: main()
    ├── parse_args()                     解析命令行
    ├── _validate_args()                 检查业务约束
    ├── set_seed()                       设置随机种子
    ├── resolve_device()                 解析 CPU/CUDA
    ├── _prepare_run_directory()         防误覆盖并准备产物目录
    ├── make_train_val_loaders()         只构造 train/validation
    ├── save_config()                    保存实际配置
    ├── build_model()                    构造 BasicCNN
    ├── CrossEntropyLoss                 构造目标函数
    ├── AdamW + CosineAnnealingLR        构造优化器与调度器
    └── 对每个 epoch
        ├── train_one_epoch()             前向、反向、更新参数
        ├── evaluate(validation_loader)   无梯度验证
        ├── append_history()              追加一行 CSV
        ├── scheduler.step()              更新下一轮学习率
        ├── save_checkpoint(last.pt)      总是保存最后状态
        └── 若 val accuracy 提高
            └── save_checkpoint(best.pt)  保存验证集最佳状态
    ↓
plot_history()                           自动生成曲线
```

注意：这条链中没有 `make_test_loader()`。训练日志还会打印 `test=not_loaded`，把协议变成
可观察证据。

最终评估是另一条独立控制流：

```text
python -m project2_cifar10.evaluate runs/cifar10/cnn/best.pt
    ↓
load_checkpoint()
    ↓
检查 selection_metric == validation_accuracy
    ↓
make_test_loader()                       此时才加载 official test
    ↓
build_model() + load_state_dict()
    ↓
engine.evaluate(test_loader)
    ↓
只报告 test，不更新 checkpoint
```

一个 batch 的数据流：

```text
PIL image / ndarray
    ↓ train transform
images: (B,3,32,32), float32
labels: (B,), int64
    ↓ BasicCNN
logits: (B,10)
    ↓ CrossEntropyLoss
loss: 标量
    ↓ backward
parameter.grad
    ↓ AdamW.step
更新后的参数
```

## 4. 先记住五个核心契约

### 4.1 数据契约

```text
单张 image:  (3,32,32), torch.float32
单个 label:  标量,       torch.int64
一批 images: (B,3,32,32), torch.float32
一批 labels: (B,),        torch.int64
```

标签不是 one-hot。`CrossEntropyLoss` 接收 `(B,10)` logits 和 `(B,)` 类别索引。

### 4.2 模型契约

```text
输入  images: (B,3,32,32)
输出  logits: (B,10)
```

输出不是 softmax 概率，也不是 `argmax` 后的整数类别。

### 4.3 划分契约

默认真实数据：

```text
official train 50,000
├── train      45,000：更新参数，使用随机增强
└── validation  5,000：选择 best，确定性预处理

official test 10,000：训练结束后独立评估
```

train 与 validation 索引必须：

- 互不相交；
- 并集覆盖官方训练集；
- 同一 seed 得到同一划分；
- test 索引完全不参与划分和选模。

### 4.4 训练函数契约

`train_one_epoch(...)`：

- 输入：模型、训练 DataLoader、损失函数、优化器、device；
- 输出：`{"loss": float, "accuracy": float}`；
- 副作用：修改模型参数、AdamW 状态、BatchNorm 运行统计量。

`evaluate(...)`：

- 输入：模型、评估 DataLoader、损失函数、device；
- 输出：相同指标字典；
- 副作用：不修改参数，不构建梯度图。

### 4.5 文件产物契约

```text
runs/cifar10/cnn/
├── config.json   解析后的配置、实际 device、选模指标
├── history.csv   train/validation 指标、学习率、耗时
├── curves.png    loss/accuracy 曲线
├── best.pt       validation accuracy 最佳 checkpoint
└── last.pt       最后完成 epoch 的 checkpoint
```

## 5. 必要理论一：为什么 CIFAR-10 比 MNIST 难

MNIST 通常是居中的单通道手写数字，背景简单。CIFAR-10 是 `32\times32` 彩色自然图像：

- 同类差异大，例如不同姿态、颜色和背景的狗；
- 不同类共享局部纹理，例如猫和狗都有毛发与眼睛；
- 目标可能只占画面一部分；
- 颜色与空间结构共同影响分类；
- 50,000 张训练图对较深网络仍不算大，容易过拟合。

因此 W3 不能只把输入通道从 1 改成 3，还需要：

- 更有层级的卷积特征；
- 合法的数据增强；
- validation 选模；
- weight decay 与学习率调度；
- 用曲线判断过拟合和优化状态。

## 6. `__init__.py`：定义包的公开入口

文件共 5 行。

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L1 | 模块 docstring | 说明这是 CIFAR-10 CNN 工程，并强调 validation-only model selection。docstring 可被 `help()` 等工具读取。 |
| L3 | `from .models import BasicCNN, build_model` | 从当前包的 `models.py` 导入模型类和工厂函数。前导点表示相对导入。 |
| L5 | `__all__ = ["BasicCNN", "build_model"]` | 声明包希望对外公开的名字。它主要影响 `from project2_cifar10 import *`，不限制显式导入。 |

包内的 `train.py` 仍直接从 `.models` 导入，因为依赖真实定义位置更清楚；`__init__.py`
主要给包使用者提供简洁 API。

## 7. `data.py`：划分、增强和测试隔离

### 7.1 文件职责

这个文件解决七件事：

1. 定义 CIFAR-10 归一化常量；
2. 分离随机训练预处理与确定性评估预处理；
3. 提供无需联网的 `SyntheticCifar10`；
4. 生成可复现、互斥的 train/validation 索引；
5. 验证 limit、batch size、worker 等输入；
6. 构造 train/validation loaders，但不接触 test；
7. 用独立函数构造 test loader。

### 7.2 L1–13：导入与常量

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L1 | 模块 docstring | 用一句话概括数据划分、增强和离线 synthetic 三项职责。 |
| L3 | `from __future__ import annotations` | 延迟求值类型注解，使 `str \| Path`、`tuple[list[int], ...]` 主要作为说明，而不在定义时急着解析。 |
| L5 | `from pathlib import Path` | 使用面向对象路径，避免手工处理 Windows/Linux 路径分隔符。 |
| L7 | `import torch` | 创建 Tensor、局部随机生成器并查询 CUDA。 |
| L8 | `DataLoader, Dataset, Subset` | `Dataset` 定义单样本访问；`DataLoader` 负责 batch/shuffle；`Subset` 用索引包装原数据，不复制全部图片。 |
| L9 | `datasets, transforms` | `datasets.CIFAR10` 提供官方数据；`transforms` 定义单样本预处理。 |
| L12 | `CIFAR10_MEAN = (...)` | 三个数按 RGB 通道给出均值。元组顺序必须与输入通道顺序一致。 |
| L13 | `CIFAR10_STD = (...)` | 三个数按 RGB 通道给出标准差。 |

按通道标准化：

$$
x'_{c,h,w}
=
\frac{x_{c,h,w}-\mu_c}{\sigma_c}.
$$

`ToTensor()` 先把常见 8 位图像缩放到 $[0,1]$，Normalize 后数值可以小于 0 或大于 1，
这是正常现象。

### 7.3 L16–33：`build_transforms`

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L16 | `def build_transforms():` | 定义无参数辅助函数。返回类型未显式注解，但实际返回两个 Compose 对象。 |
| L17 | docstring | 明确 train 是 stochastic，evaluation 是 deterministic。这不是风格描述，而是实验契约。 |
| L19 | `train_transform = transforms.Compose(` | Compose 把列表中的变换按顺序串成单个可调用对象。 |
| L20 | `[` | 开始有序 transform 列表；顺序会影响结果。 |
| L21 | `RandomCrop(32, padding=4)` | 先在四周补 4 像素，再随机裁出 `32\times32`，等价于制造小范围平移和边界变化。 |
| L22 | `RandomHorizontalFlip()` | 默认以 0.5 概率水平翻转。对 CIFAR-10 类别通常保持语义。 |
| L23 | `ToTensor()` | 将 PIL/ndarray 变为 `(C,H,W)` float Tensor，并缩放像素。它必须早于 Normalize。 |
| L24 | `Normalize(MEAN, STD)` | 对三个通道分别标准化。 |
| L25–26 | `]` 与 `)` | 闭合列表和 Compose 构造；它们与 L19 构成完整表达式。 |
| L27 | `evaluation_transform = transforms.Compose(` | 单独构造验证/test 预处理，避免随机增强进入评估。 |
| L28–31 | ToTensor + Normalize | 与训练保持相同数值尺度，但不 crop、不 flip。 |
| L32 | `)` | 闭合 evaluation Compose。 |
| L33 | `return train_transform, evaluation_transform` | 返回二元组，调用者用解包分别取得。 |

#### 为什么增强只能用于 train

训练增强是在表达一个先验：

$$
f(T(x)) \approx f(x),
$$

其中 $T$ 是不改变标签语义的变换。如果 validation 每次也随机变换：

- 同一 checkpoint 每次面对的评估样本不同；
- 指标方差变大；
- 两个 checkpoint 的比较不再完全公平；
- best 可能由随机裁剪的“运气”决定。

增强也不是越多越好。水平翻转适合猫、狗、汽车等，但数字 6 的旋转可能改变语义，医学
影像也可能有方向约束。

### 7.4 L36–71：`SyntheticCifar10`

这个 Dataset 生成十种带颜色和位置规则的 `3\times32\times32` 图案。目的不是模拟真实
CIFAR-10，而是在断网时验证 Dataset、CNN、反向传播、日志和 checkpoint。

#### L36–42：类声明与边界

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L36 | `class SyntheticCifar10(Dataset):` | 继承 PyTorch Dataset 协议，承诺实现 `__len__` 和 `__getitem__`。 |
| L37–42 | 类 docstring | 说明图像 shape、用途和局限，并强调 train/validation/test 使用独立 seed。不能把 synthetic accuracy 写成真实结果。 |

#### L44–65：初始化逐行

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L44 | `def __init__(self, size: int, seed: int = 0):` | 构造实例时自动调用；`self` 是当前对象；`size` 必传，`seed` 默认 0。 |
| L45–46 | size 检查 | 非正样本数没有训练意义，还会导致指标除零，因此尽早抛 `ValueError`。 |
| L48 | 局部 `torch.Generator` | 创建只服务本数据集的随机生成器。同一 seed 得到同样图案，不必重置全局随机状态。 |
| L49 | `arange(size, int64) % 10` | 生成循环标签 0–9；`int64` 满足交叉熵类别索引要求。 |
| L50 | `labels[randperm(...)]` | 使用同一局部 generator 随机打乱标签，并保存为实例状态。 |
| L51 | `torch.zeros(size, 3, 32, 32)` | 一次创建所有黑色 RGB 图像，shape 顺序是 `N,C,H,W`。 |
| L53 | `for index, label_tensor in enumerate(...)` | 遍历每个样本；enumerate 同时返回样本位置与标签 Tensor。 |
| L54 | `label = int(label_tensor)` | 把 0 维 Tensor 转成 Python int，便于取模、整除和索引。 |
| L55 | `channel = label % 3` | 十类循环映射到 RGB 三个主通道。 |
| L56 | `row = 2 + (label // 5) * 16` | 类别 0–4 在上半区，5–9 在下半区。`//` 是整除。 |
| L57 | `column = 2 + (label % 5) * 6` | 每排五个类别，用取余决定横向位置。 |
| L58 | 四维切片赋值为 1 | 在主通道画 `10\times5` 亮块；Python 切片右端不包含。 |
| L59 | 第二通道条带赋值为 0.45 | 再加入跨宽度的次级颜色条，让图案不只是单个亮块。`:` 表示取该维全部列。 |
| L61 | `0.04 * torch.randn(...)` | 生成同 shape 高斯噪声，标准差缩为 0.04，并继续使用局部 generator。 |
| L62 | 加噪并 `clamp(0,1)` | 截断到合法像素范围，返回新 Tensor。 |
| L63 | mean Tensor + `view(1,3,1,1)` | 将三通道均值变成可沿 N/H/W 广播的 shape。 |
| L64 | std 同理 | 得到 `(1,3,1,1)`，与图像 `(N,3,32,32)` 广播。 |
| L65 | `self.images = (images-mean)/std` | 使用与真实 CIFAR-10 一样的归一化尺度，并保存到实例。 |

广播时：

```text
images: (N,3,32,32)
mean:   (1,3, 1, 1)
std:    (1,3, 1, 1)
结果:   (N,3,32,32)
```

#### L67–71：Dataset 协议

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L67 | `def __len__(self) -> int:` | 定义 `len(dataset)` 的行为。 |
| L68 | `return len(self.labels)` | 样本数由标签长度决定；图像与标签必须一一对应。 |
| L70 | `def __getitem__(self, index: int):` | 定义 `dataset[index]`。DataLoader 会反复调用它。 |
| L71 | 返回 `(image,label)` | 单样本 image 是 `(3,32,32)`，DataLoader 堆叠后增加 batch 维。 |

### 7.5 L74–90：固定 train/validation 划分

`split_train_validation_indices` 只处理索引，不读取图片。它的函数契约：

```python
split_train_validation_indices(dataset_size, validation_size, seed)
    -> (train_indices, validation_indices)
```

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L74–78 | 多行函数签名 | 三个整数输入，返回两个整数列表组成的元组。多行尾逗号便于维护。 |
| L79 | docstring | 三个关键词是 deterministic、disjoint、official training set。 |
| L81–82 | `dataset_size <= 1` | 至少要有两个样本才能分出两个非空集合。 |
| L83–84 | 链式比较 | `0 < validation_size < dataset_size` 等价于两个条件同时成立；否则抛明确错误。 |
| L86 | 局部 generator | 同 seed 控制划分，不依赖其他代码消耗了多少全局随机数。 |
| L87 | `randperm(...).tolist()` | 得到 0 到 `dataset_size-1` 的无重复随机排列，再转 Python list。 |
| L88 | 前 `validation_size` 个索引 | 切片构成 validation。 |
| L89 | 剩余索引 | 从切分点到末尾全部作为 train。 |
| L90 | 返回 train 在前、validation 在后 | 顺序与调用方变量解包一致。 |

#### 为什么必须先划 validation

机器学习实验包含两类不同决策：

- **参数学习**：梯度用 train 更新权重；
- **模型/超参数选择**：用 validation 决定 epoch、学习率、结构和 checkpoint。

test 的职责是估计“所有选择完成后，这个方案对未见数据怎样”。若反复看 test 再决定
checkpoint，相当于让 test 信息通过人为决策泄漏进模型。

形式上，假设在 $M$ 个候选 checkpoint 中选择：

$$
\hat m
=
\arg\max_{m\in\{1,\dots,M\}}
\operatorname{Acc}_{val}(m).
$$

最后只计算：

$$
\operatorname{Acc}_{test}(\hat m).
$$

test 不出现在 $\arg\max$ 内。

### 7.6 L93–113：三个防御式辅助函数

#### `_validate_limit`

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L93 | 函数签名 | `name` 用于错误消息，`limit` 允许 int 或 None；前导下划线表示模块内部辅助函数。 |
| L94 | `limit is not None and limit <= 0` | None 表示“不限制”；只对显式整数检查正数。`and` 短路保证 None 不参与大小比较。 |
| L95 | f-string ValueError | 把具体参数名写进错误消息，便于定位是 train、val 还是 test limit。 |

#### `_limit_indices`

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L98 | 签名 | 输入索引列表与可选上限，返回列表。 |
| L99–100 | None 分支 | 不限制时原样返回同一个列表对象，不做无意义复制。 |
| L101 | 安全切片 | 上限取 limit 与列表长度较小值；即使不取 min，Python 大切片也安全，这里把意图写得更明确。 |

#### `_loader_options`

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L104 | 签名 | 把两个 loader 共用选项集中成字典。 |
| L105–106 | batch size 检查 | batch 必须为正。 |
| L107–108 | worker 检查 | worker 数允许 0，不允许负数。Windows 初学阶段用 0 最稳。 |
| L109–113 | 返回字典 | 保存 batch size、worker 数与 pin memory；后面用 `**options` 展开为关键字参数。 |
| L112 | `torch.cuda.is_available()` | 有 CUDA 时启用页锁定内存，配合 `non_blocking=True` 可能加速 CPU→GPU 传输。 |

### 7.7 L116–179：只构造 train/validation

#### 函数签名和入口检查

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L116–125 | 多行签名 | 接收数据名、目录、batch size、validation 大小、worker、seed 和两个 quick-run 上限；返回两个 DataLoader。 |
| L126 | docstring | `without touching the test split` 是最重要的不变量。 |
| L128 | `options = _loader_options(...)` | 先验证 loader 参数并得到公共字典。 |
| L129–130 | 验证两个 limit | 防止 0/负数产生空数据或被 `or default` 静默回退。 |
| L131 | `dataset_name.lower()` | 将 `CIFAR10` 等大小写形式规范为小写并重新赋值。 |

#### 真实 CIFAR-10 分支

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L133 | `if dataset_name == "cifar10":` | 进入真实数据分支；`==` 比较字符串值。 |
| L134 | 解包两套 transform | train 与 evaluation 必须绑定不同预处理。 |
| L135 | `Path(data_dir)` | 统一字符串/Path 输入。 |
| L136–141 | `augmented_dataset` | 构造官方 `train=True` 数据，缺失时下载，每次取样应用随机 train transform。变量注解为 Dataset。 |
| L142–147 | `evaluation_dataset` | 仍然是同一官方 `train=True` 数据，但绑定确定性 evaluation transform。不是 test set。 |
| L148–150 | 调划分函数 | 以官方训练集长度、validation size、seed 生成一对互斥索引。 |
| L151–154 | train `Subset` | 用 train 索引包装 augmented dataset，再可选截取 quick-run 数量。图片本身没有被复制。 |
| L155–158 | validation `Subset` | 用 validation 索引包装 evaluation dataset，因此同一底层训练数据在两个集合使用不同 transform。 |

为什么要创建两个 `datasets.CIFAR10(train=True)` 对象？因为 transform 是 Dataset 的属性。
若只建一个对象再套两个 Subset，两边会共享同一个 transform，无法做到“train 随机、val
确定”。两个对象都指向磁盘上的同一数据文件，不等于复制两份 50,000 张图片。

#### synthetic 与错误分支

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L159 | `elif dataset_name == "synthetic":` | 真实分支未命中时检查离线数据。 |
| L160 | train synthetic | 显式 limit 优先，否则默认 200；seed 使用原值。前面已拒绝 0，所以 `or` 不会吞掉非法 0。 |
| L161 | validation synthetic | 默认 80，seed 加 1，使噪声实例独立但生成规则相同。 |
| L162–165 | else + ValueError | 对未知数据名快速失败；`!r` 用 repr 展示输入，保留引号和异常空格。 |

#### DataLoader 构造

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L167 | loader generator | 单独控制训练 shuffle 顺序，提高同 seed 下的可复现性。 |
| L168–173 | train loader | 训练使用 `shuffle=True`，传入局部 generator，并用 `**options` 展开公共参数。 |
| L174–178 | validation loader | 验证不打乱。完整遍历时顺序不影响平均指标，固定顺序更利于调试。 |
| L179 | 返回二元组 | 调用方解包为 `train_loader, validation_loader`。 |

### 7.8 L182–216：独立 `make_test_loader`

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L182–189 | 多行签名 | 与训练 loader 分开设计；只返回一个 DataLoader。`limit_test` 仅用于 quick evaluation。 |
| L190 | docstring | 强调 held-out 和 independent evaluation entry point。 |
| L192 | 公共 loader 选项 | 同时完成 batch/worker 检查。 |
| L193 | 验证 test limit | 非正值尽早失败。 |
| L194 | 数据名小写化 | 与训练接口保持一致。 |
| L196 | 真实 CIFAR 分支 | 只有独立评估入口才应调用这里。 |
| L197 | `_, evaluation_transform = ...` | 丢弃 train transform；下划线表示有意忽略。 |
| L198–203 | `datasets.CIFAR10(train=False)` | 这里才加载官方 10,000 张 test，使用确定性 transform。 |
| L204–208 | 可选 test Subset | quick evaluation 只暴露前若干 test 索引；`range` 不复制图片。 |
| L209–210 | synthetic test | 默认 80，seed 加 2，与 train/validation 实例独立。 |
| L211–214 | 非法名称错误 | 与训练函数保持一致的快速失败。 |
| L216 | 构造并返回 test loader | `shuffle=False`，展开公共选项。 |

### 7.9 立即验证

```powershell
python -c "from project2_cifar10.data import SyntheticCifar10; d=SyntheticCifar10(23,7); print(len(d), d[0][0].shape, d[0][0].dtype, d[0][1].dtype)"
```

预期：

```text
23 torch.Size([3, 32, 32]) torch.float32 torch.int64
```

划分不变量：

```powershell
python -c "from project2_cifar10.data import split_train_validation_indices as s; a,b=s(100,20,7); print(len(a),len(b),len(set(a)&set(b)),len(set(a)|set(b)))"
```

预期 `80 20 0 100`。

## 8. `models.py`：从局部特征到全局分类

### 8.1 L1–3：导入

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L1 | 模块 docstring | 说明这是面向 `32\times32` RGB 分类的 baseline CNN。 |
| L3 | `from torch import nn` | 导入 `nn.Module`、层和容器。 |

### 8.2 先理解二维卷积

忽略 batch，输入：

$$
X\in\mathbb R^{C_{in}\times H\times W},
$$

卷积核：

$$
K\in\mathbb R^{C_{out}\times C_{in}\times K_h\times K_w}.
$$

第 $o$ 个输出通道：

$$
Y_{o,i,j}
=
b_o+
\sum_{c=1}^{C_{in}}
\sum_{u=1}^{K_h}
\sum_{v=1}^{K_w}
K_{o,c,u,v}
X_{c,iS_h+u-P_h,jS_w+v-P_w}.
$$

PyTorch 的 `Conv2d` 严格说计算互相关，没有翻转卷积核；但核由训练学习，对作为可学习
局部算子没有实质影响。

单个空间维度的输出：

$$
H_{out}
=
\left\lfloor
\frac{H_{in}+2P-D(K-1)-1}{S}+1
\right\rfloor.
$$

本模型所有卷积使用 $K=3,P=1,S=1,D=1$，所以高宽不变。`MaxPool2d(2)` 默认
$K=2,S=2,P=0$，所以高宽减半。

无 bias 卷积参数量：

$$
C_{out}C_{in}K_hK_w.
$$

它与输入高宽无关，因为同一组权重在所有空间位置共享；但计算量仍随
$H_{out}W_{out}$ 增长。

### 8.3 L6–38：`BasicCNN.__init__` 逐行

#### 类、父类和第一阶段

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L6 | `class BasicCNN(nn.Module):` | 所有 PyTorch 模型继承 `nn.Module`，获得参数注册、`.to()`、`.train()`、`state_dict()` 等能力。 |
| L7 | 类 docstring | 概括“三个卷积阶段 + 全局平均池化”。 |
| L9 | `__init__(..., num_classes=10)` | 构造模型结构；类别数可配置但默认 CIFAR-10 的 10。 |
| L10 | `super().__init__()` | 初始化父类。漏掉会破坏子模块和参数注册。 |
| L11 | `self.features = nn.Sequential(` | 按顺序注册特征提取层。赋给 `self` 后，PyTorch 才能递归发现参数。 |
| L12 | `Conv2d(3,32,3,padding=1,bias=False)` | RGB 3 通道映射到 32 个特征通道，高宽保持 32。关闭 bias，因为后接 BatchNorm 的平移参数会承担类似作用。 |
| L13 | `BatchNorm2d(32)` | 对 32 通道分别标准化，学习 32 个 $\gamma$ 与 32 个 $\beta$，并维护 running mean/variance。 |
| L14 | `ReLU(inplace=True)` | 逐元素计算 $\max(0,x)$；inplace 尽量复用激活内存。不要在复杂残差/多分支图中盲目使用原地操作。 |
| L15 | `Conv2d(32,32,...)` | 在同一分辨率继续组合局部模式，输出仍为 `(B,32,32,32)`。 |
| L16–17 | BN + ReLU | 第二次归一化和非线性。连续两个 `3\times3` 卷积的感受野变为 `5\times5`。 |
| L18 | `MaxPool2d(2)` | 对每个 `2\times2` 窗口取最大值，高宽从 32 降为 16，通道不变。 |

#### 第二、第三阶段与全局聚合

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L19 | `Conv2d(32,64,...)` | 通道扩为 64，高宽保持 16；更深层组合更多低层特征。 |
| L20–21 | BN64 + ReLU | 对 64 通道归一化并引入非线性。 |
| L22 | `Conv2d(64,64,...)` | 在 `16\times16` 分辨率继续提取特征。 |
| L23–24 | BN64 + ReLU | 第二阶段的归一化和激活。 |
| L25 | 第二次池化 | `16\times16 → 8\times8`。 |
| L26 | `Conv2d(64,128,...)` | 通道扩到 128，空间保持 `8\times8`。 |
| L27–28 | BN128 + ReLU | 第三阶段第一组归一化和激活。 |
| L29 | `Conv2d(128,128,...)` | 最后一层卷积继续组合高层特征。 |
| L30–31 | BN128 + ReLU | 最后一组归一化和激活。 |
| L32 | `AdaptiveAvgPool2d((1,1))` | 对每个通道的整个空间特征图求平均，任何输入空间尺寸都输出 `1\times1`。 |
| L33 | `)` | 闭合 features Sequential。 |
| L34 | `self.classifier = nn.Sequential(` | 定义分类头，与特征提取器分离。 |
| L35 | `Flatten()` | `(B,128,1,1) → (B,128)`，保留 batch 维。 |
| L36 | `Dropout(p=0.3)` | 训练时随机将 30% 特征置零并缩放保留值；评估时关闭。无可训练参数。 |
| L37 | `Linear(128,num_classes)` | 产生十个 logits；权重 `(10,128)`，bias `(10,)`。 |
| L38 | `)` | 闭合 classifier。 |

### 8.4 L40–41：前向传播

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L40 | `def forward(self, images):` | 定义前向。一般写 `model(images)`，不要直接调 `forward`，因为 `nn.Module.__call__` 还处理 hooks 等机制。 |
| L41 | `return self.classifier(self.features(images))` | 先提取特征，再送分类头；嵌套调用省去中间变量，返回 `(B,10)` logits。 |

完整 shape 流：

```text
(B,3,32,32)
→ Conv/BN/ReLU → (B,32,32,32)
→ Conv/BN/ReLU → (B,32,32,32)
→ MaxPool      → (B,32,16,16)
→ Conv/BN/ReLU → (B,64,16,16)
→ Conv/BN/ReLU → (B,64,16,16)
→ MaxPool      → (B,64, 8, 8)
→ Conv/BN/ReLU → (B,128,8,8)
→ Conv/BN/ReLU → (B,128,8,8)
→ GAP          → (B,128,1,1)
→ Flatten      → (B,128)
→ Dropout/Linear → (B,10)
```

### 8.5 参数量逐项计算

| 模块 | 参数量 |
|---|---:|
| Conv 3→32 | $32\times3\times3\times3=864$ |
| BN32 | $32+32=64$ |
| Conv 32→32 | $32\times32\times3\times3=9{,}216$ |
| BN32 | $64$ |
| Conv 32→64 | $18{,}432$ |
| BN64 | $128$ |
| Conv 64→64 | $36{,}864$ |
| BN64 | $128$ |
| Conv 64→128 | $73{,}728$ |
| BN128 | $256$ |
| Conv 128→128 | $147{,}456$ |
| BN128 | $256$ |
| Linear 128→10 | $128\times10+10=1{,}290$ |
| 总计 | **288,746** |

`bias=False` 使卷积没有 `+C_out` 项。BatchNorm 的 running mean/variance 是 buffer，会进入
`state_dict`，但不是梯度更新的可训练参数，所以不计入上表可训练参数量。

### 8.6 感受野和分层表征

设第 $l$ 层相邻输出在原图上的跳跃为 $j_l$，感受野为 $r_l$：

$$
j_l=j_{l-1}s_l,
\qquad
r_l=r_{l-1}+(k_l-1)j_{l-1}.
$$

从 $r_0=1,j_0=1$ 开始，本模型最后一层卷积单元的理论感受野达到 `32×32`。直观上：

- 低层学习颜色边缘和小纹理；
- 中层组合成局部形状；
- 深层可以整合接近整幅图的信息；
- global average pooling 再把每个通道的空间响应聚合为一个数。

### 8.7 BatchNorm、Dropout 与模式

BatchNorm 训练时用当前 batch 均值/方差并更新运行统计；评估时使用运行统计。Dropout
训练时随机置零，评估时关闭。因此：

- `model.train()` 决定它们按训练模式工作；
- `model.eval()` 决定它们按评估模式工作；
- 是否记录梯度由 autograd 上下文控制，是另一件事。

### 8.8 L44–49：模型工厂

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L44 | `def build_model(name: str = "cnn") -> nn.Module:` | 输入模型名，返回 Module；默认 cnn。工厂函数让 train/evaluate 使用同一构造入口。 |
| L45 | docstring | 明确集中定义是为了共享模型结构。 |
| L47 | `if name.lower() == "cnn":` | 允许大小写差异。 |
| L48 | `return BasicCNN()` | 每次返回一个全新、随机初始化的模型实例。 |
| L49 | `raise ValueError(...)` | 未知模型名快速失败；`!r` 保留输入表示。 |

立即验证：

```powershell
python -c "import torch; from project2_cifar10.models import build_model; m=build_model('cnn'); print(m(torch.randn(4,3,32,32)).shape); print(sum(p.numel() for p in m.parameters() if p.requires_grad))"
```

预期：

```text
torch.Size([4, 10])
288746
```

## 9. `engine.py`：训练与评估的核心状态机

### 9.1 L1–14：导入与 device 解析

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L1 | 模块 docstring | 表明这里提供可复用的训练/评估循环，不绑定 CIFAR-10 或具体模型。 |
| L3 | future annotations | 延迟类型注解求值。 |
| L5 | `import torch` | 用于 device、inference mode 等核心功能。 |
| L8 | `resolve_device(requested: str)` | 输入 `auto/cpu/cuda/cuda:0` 等字符串，返回 `torch.device`。 |
| L9 | `if requested == "auto":` | 只有精确字符串 auto 才自动选择。 |
| L10 | 条件表达式 | CUDA 可用就构造 cuda device，否则 cpu；`a if condition else b` 是一行选择表达式。 |
| L11 | `device = torch.device(requested)` | 对显式请求进行解析，例如 `cuda:0`。 |
| L12 | CUDA 可用性检查 | 用户明确要求 CUDA 但环境不可用时，不应静默退回 CPU。`device.type` 只看大类。 |
| L13 | `RuntimeError` | 错误来自运行环境不满足请求，比 ValueError 更贴切。 |
| L14 | `return device` | 返回验证后的 device。 |

模型和参与同一算子的 Tensor 必须在同一 device：

```python
model = model.to(device)
images = images.to(device)
labels = labels.to(device)
```

`model.to(device)` 能移动所有已注册参数和 buffer，这正是模型层必须赋给 `self` 的另一个
原因。

### 9.2 L17–43：`train_one_epoch`

一个 epoch 表示训练 DataLoader 被完整遍历一次。若训练集有 $N$ 个样本、batch size
为 $B$，大约执行：

$$
\left\lceil\frac{N}{B}\right\rceil
$$

次参数更新。

#### 模式与累加器

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L17 | 函数签名 | 输入模型、loader、loss、optimizer、device；返回字符串到 float 的字典。 |
| L18 | `model.train()` | 开启训练模式：Dropout 生效，BatchNorm 使用当前 batch 并更新 running stats。它不等于“开启 autograd”。 |
| L19 | `loss_sum = 0.0` | 累计所有样本 loss 的总和，使用 Python float 避免保留计算图。 |
| L20 | `correct = 0` | 累计正确预测数量。 |
| L21 | `sample_count = 0` | 累计真实样本数，处理最后一批不足 batch size 的情况。 |

#### 每个 batch 的训练五步

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L23 | `for images, labels in dataloader:` | DataLoader 产出二元 batch，序列解包为图像和标签。 |
| L24 | images 移 device | `.to()` 通常返回新 Tensor，因此必须重新赋值。`non_blocking=True` 在 pinned memory + CUDA 时才可能异步。 |
| L25 | labels 移 device | logits 和 labels 必须同设备，标签仍保持 int64。 |
| L27 | `zero_grad(set_to_none=True)` | 清除旧梯度。设为 None 通常比填零少一次内存写入；下次 backward 会创建新 grad。 |
| L28 | `logits = model(images)` | 前向传播并建立动态计算图，输出 `(B,10)`。 |
| L29 | `loss_fn(logits, labels)` | 交叉熵将一批预测和标签归约为标量，默认 reduction 是 mean。 |
| L30 | `loss.backward()` | 从标量 loss 的梯度 1 出发，沿计算图反向应用链式法则，将梯度写入参数 `.grad`。 |
| L31 | `optimizer.step()` | AdamW 读取参数与梯度，更新参数和自身动量状态；它不会自动清梯度。 |

因果顺序：

```text
清旧梯度 → 用当前参数前向 → 算 loss → 对当前参数求梯度 → 更新参数
```

为什么 PyTorch 默认累加梯度？因为有些任务需要多个 micro-batch 累积后再 step。但本项目
每个 batch 都更新，所以每轮 forward 前先清旧梯度。

#### 整轮指标

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L33 | `batch_size = labels.shape[0]` | 使用当前 batch 的真实样本数；最后一批可能更小。 |
| L34 | `loss.item() * batch_size` | `.item()` 把标量 Tensor 转 Python 数；默认 loss 是 batch 平均，乘 batch size 还原该批总和。 |
| L35 | accuracy 累加 | 每行最大 logit 对应预测类别；比较标签得到 bool Tensor，再 sum 成正确数。 |
| L36 | 样本计数 | 累计分母。 |
| L38–39 | 空 loader 检查 | 防止除零，并把数据管道错误尽早暴露。 |
| L40–43 | 返回字典 | loss 和 accuracy 都按全体样本平均，而不是简单平均 batch 平均值。 |

正确的整轮平均 loss：

$$
\bar L
=
\frac{\sum_b n_b L_b}{\sum_b n_b}.
$$

若一个 batch 有 128 个样本、最后一个只有 16 个，直接平均两个 batch loss 会让 16 个
样本获得与 128 个样本相同的权重。

### 9.3 交叉熵为什么直接接 logits

模型给出 logits $z_1,\dots,z_C$。softmax：

$$
p_k=\frac{e^{z_k}}{\sum_j e^{z_j}}.
$$

真实类别为 $y$ 时：

$$
L=-\log p_y
=
-z_y+\log\sum_j e^{z_j}.
$$

梯度：

$$
\frac{\partial L}{\partial z_k}
=
p_k-\mathbb 1[k=y].
$$

`CrossEntropyLoss` 内部组合 log-softmax 和负对数似然，并使用数值稳定的 log-sum-exp。
所以模型末尾不要手动加 softmax。

### 9.4 L46–69：`evaluate`

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L46 | 函数签名 | 与训练函数相似，但没有 optimizer，因为评估不能更新参数。 |
| L47 | `model.eval()` | 关闭 Dropout，BatchNorm 改用训练期累计的 running stats。 |
| L48–50 | 三个累加器 | 与训练保持同一指标口径。 |
| L52 | `with torch.inference_mode():` | 在缩进块内关闭 autograd 记录并启用额外推理优化，退出后恢复上下文。 |
| L53 | 遍历评估 loader | 可以是 validation，也可以是独立 test；engine 不关心集合语义。 |
| L54–55 | 数据移 device | 与训练一样保证模型/输入/标签同设备。 |
| L56 | 前向 logits | 不构建反向图。 |
| L57 | 计算 loss | 用于评估，不调用 backward。 |
| L59–62 | 累加指标 | 与训练完全同口径，保证曲线可比较。 |
| L64–65 | 空 loader 检查 | 防止无意义指标。 |
| L66–69 | 返回样本平均值 | 返回普通 Python float 字典。 |

`model.eval()` 与 `inference_mode()` 不能互相替代：

- `eval()` 改变模块行为；
- `inference_mode()` 改变 autograd 是否记录运算。

### 9.5 立即验证参数更新

工程测试会 clone 训练前参数，跑一轮，再验证至少一个参数不再 `allclose`。只看到 loss
数字并不能证明优化器真的更新了参数。

## 10. `utils.py`：随机性、JSON、CSV 与 checkpoint

### 10.1 L1–11：导入

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L1 | 模块 docstring | 列出 reproducibility、CSV、configuration、checkpoint 四项职责。 |
| L3 | future annotations | 同前。 |
| L5 | `import csv` | 使用标准库读写 CSV，不引入 pandas。 |
| L6 | `import json` | 把普通配置保存为人可读文本。 |
| L7 | `import random` | Python 标准库随机生成器。 |
| L8 | `Path` | 路径处理。 |
| L10 | `numpy as np` | 使用约定别名 np，设置 NumPy seed。 |
| L11 | `torch` | 设置 PyTorch seed、保存/加载状态。 |

### 10.2 L14–19：`set_seed`

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L14 | `def set_seed(seed: int) -> None:` | 只产生副作用，不返回有意义的值。 |
| L15 | `random.seed(seed)` | 控制 Python 标准库随机过程。 |
| L16 | `np.random.seed(seed)` | 控制 NumPy 旧式全局随机状态。 |
| L17 | `torch.manual_seed(seed)` | 控制 PyTorch CPU 等随机状态。 |
| L18 | CUDA 可用判断 | CPU 环境不调用 CUDA API。 |
| L19 | `manual_seed_all` | 给所有 CUDA 设备设置 seed。 |

固定 seed 只减少随机差异，不保证跨 GPU、库版本、并行算法逐 bit 一致。更严格确定性可能
需要 `torch.use_deterministic_algorithms(True)`，并接受性能下降或算子限制。科研结论也
不应只依赖一个幸运 seed。

### 10.3 L22–28：`save_config`

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L22 | 函数签名 | 输入路径和普通字典，副作用是写 JSON。 |
| L23 | `path = Path(path)` | 统一字符串与 Path。 |
| L24 | `mkdir(parents=True, exist_ok=True)` | 递归创建父目录，已存在不报错。 |
| L25 | `path.write_text(` | 一次写入文本并自动打开/关闭文件。 |
| L26 | `json.dumps(...)` | 将字典序列化；`ensure_ascii=False` 保留中文；`indent=2` 便于阅读；`sort_keys=True` 稳定键顺序。 |
| L27 | `encoding="utf-8"` | 明确编码，避免 Windows 默认编码差异。 |
| L28 | `)` | 完成写盘。 |

为什么 JSON 还要保存一份，checkpoint 里不是也有 config 吗？因为 JSON 不需要 PyTorch
就能直接查看、做 diff 或被脚本读取；checkpoint config 则让状态文件自身完整。

### 10.4 L31–57：`save_checkpoint`

#### 函数签名与临时路径

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L31–39 | 多行签名 | 输入路径、模型、优化器、可选 scheduler、epoch、历史最佳 val accuracy 和配置；不返回值。 |
| L40 | Path 统一 | 同前。 |
| L41 | 创建父目录 | 第一次保存时目录可能不存在。 |
| L42 | `path.with_suffix(path.suffix + ".tmp")` | 将 `best.pt` 临时变为 `best.pt.tmp`，先写完整临时文件。 |

#### checkpoint 字典

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L43 | `torch.save(` | 开始序列化状态字典。 |
| L44 | `{` | 开始命名字段，避免依赖元组位置。 |
| L45 | `model.state_dict()` | 保存参数和持久 buffer；BatchNorm running mean/variance 也在其中。 |
| L46 | `optimizer.state_dict()` | 保存 AdamW step、一阶/二阶矩与参数组。恢复训练不可缺。 |
| L47–49 | scheduler 条件表达式 | 有 scheduler 就保存状态，否则保存 None，使 checkpoint schema 稳定。 |
| L50 | `epoch` | 记录状态来自哪一轮。 |
| L51 | `best_val_accuracy` | 记录截至当前轮的历史最佳验证准确率。 |
| L52 | 固定 selection metric | 把“best 由什么选出”写进文件，防止未来误把 test-selected checkpoint 当合规结果。 |
| L53 | `config` | 保存构造此次实验的配置。 |
| L54 | `}` | 闭合字典。 |
| L55 | `temporary_path` | 先写临时目标，而不是直接覆盖正式 checkpoint。 |
| L56 | `)` | 完成临时文件序列化。 |
| L57 | `temporary_path.replace(path)` | 临时文件完整写成后替换正式路径，降低中断时把旧 checkpoint 写坏的风险。 |

这里通常称“原子式替换”思路：真正的原子性仍受操作系统和文件系统保证影响，但比直接向
`best.pt` 写一半后中断安全得多。

### 10.5 L60–65：`load_checkpoint`

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L60 | 函数签名 | 输入文件路径和目标 device，返回 checkpoint 字典。 |
| L61 | `return torch.load(` | 直接返回反序列化结果。 |
| L62 | `Path(path)` | 统一路径。 |
| L63 | `map_location=device` | GPU 保存的 Tensor 也可映射到 CPU，或映射到指定 GPU。 |
| L64 | `weights_only=True` | 使用受限反序列化模式。名字不表示只返回模型权重；基础字典、Tensor、optimizer 状态仍可读取。 |
| L65 | `)` | 完成加载调用。 |

### 10.6 L68–76：`append_history`

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L68 | 签名 | `row` 的键是字符串，值是 float/int；每次追加一个 epoch。 |
| L69 | Path 统一 | 同前。 |
| L70 | 创建父目录 | 允许首次写历史。 |
| L71 | `write_header = not path.exists()` | 打开前判断是否新文件；只有第一次写列名。 |
| L72 | `with path.open("a",...)` | 追加模式；with 保证关闭；UTF-8；`newline=""` 避免 Windows CSV 空行。 |
| L73 | `csv.DictWriter(..., fieldnames=list(row))` | 使用 row 字典键的插入顺序作为列顺序。 |
| L74–75 | 首次写 header | 新文件写 `epoch,train_loss,...`；旧文件不重复。 |
| L76 | `writer.writerow(row)` | 写入当前 epoch 一行。 |

## 11. `train.py`：把全部组件编排成实验

### 11.1 L1–16：导入层次

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L1 | 模块 docstring | 说明这是 validation-driven 的命令行训练入口。 |
| L3 | future annotations | 同前。 |
| L5 | `argparse` | 标准库命令行解析器。 |
| L6 | `time` | 用高精度计时器记录每轮耗时。 |
| L7 | `Path` | 路径参数和输出拼接。 |
| L9 | `torch` | optimizer、scheduler 和参数量统计。 |
| L10 | `nn` | 构造 CrossEntropyLoss。 |
| L12 | 数据入口 | 训练文件只能导入 train/validation loader，不导入 test loader。 |
| L13 | engine 三函数 | 设备、训练、评估各自复用。 |
| L14 | 模型工厂 | 训练和独立评估共享模型定义。 |
| L15 | `plot_history` | 训练结束自动画图。 |
| L16 | 四个 utils | 入口只负责编排，不复制持久化实现。 |

### 11.2 L19–50：`parse_args`

| 行号 | 参数代码 | 作用 |
|---|---|---|
| L19 | 返回 Namespace | 后面用 `args.epochs` 点号访问。 |
| L20 | `ArgumentParser` | description 会出现在 `--help`。 |
| L21 | `--model` | 当前只有 cnn，但保留模型名作为配置与输出目录维度。`choices` 提前验证。 |
| L22–27 | `--dataset` | 默认真实 cifar10；synthetic 明确只用于离线 smoke。 |
| L28 | `--data-dir` | 字符串自动转 Path，默认主仓 `data`。 |
| L29 | `--output-dir` | 输出根目录，后面还拼接模型名。 |
| L30 | `--epochs=30` | 目标训练轮数。argparse 保证 int，不保证正数。 |
| L31 | `--batch-size=128` | 每批上限；最后一批可更小。 |
| L32 | `--learning-rate=1e-3` | AdamW 初始学习率 0.001。 |
| L33 | `--weight-decay=5e-4` | 解耦权重衰减强度 0.0005。 |
| L34 | `--validation-size=5000` | 从官方 50k train 中划 5k validation。 |
| L35 | `--num-workers=0` | Windows 入门最稳定；以后可实测增大。 |
| L36 | `--device=auto` | 也允许显式 cpu、cuda、cuda:0。 |
| L37 | `--seed=42` | 控制主要随机源与划分。 |
| L38–39 | train/val limit | 无默认即 None，只用于 quick run。 |
| L40–44 | `--scheduler` | choices 为 cosine/none，默认 cosine。 |
| L45–49 | `--overwrite` | `store_true`：出现为 True；明确授权删除同 run 的生成产物。 |
| L50 | `parse_args()` | 解析当前进程命令行；非法参数时自动打印帮助并退出。 |

运行：

```powershell
python -m project2_cifar10.train --help
```

### 11.3 L53–63：业务参数检查

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L53 | `_validate_args(args) -> None` | 前导下划线表示入口内部辅助函数。 |
| L54–55 | epochs > 0 | 防止零轮训练后不存在 history/best。 |
| L56–57 | batch size > 0 | 尽早报错。 |
| L58–59 | learning rate > 0 | 负值或 0 不符合本训练定义。 |
| L60–61 | weight decay >= 0 | 允许关闭衰减的 0，不允许负衰减。 |
| L62–63 | validation size > 0 | 真实分支还会进一步检查小于数据集长度。 |

### 11.4 L66–73：把 Namespace 转成可序列化配置

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L66 | 函数签名 | 输入命令行对象和解析后的真实 device，返回普通 dict。 |
| L67 | `vars(args).copy()` | Namespace 底层字典的浅复制，避免后续转换 Path 时修改 args 自身。 |
| L68 | `for key,value in list(config.items())` | 遍历键值；先转 list，表达在遍历时可能修改字典值的安全意图。 |
| L69 | `isinstance(value, Path)` | 找出 JSON 不能直接序列化的 Path。 |
| L70 | `config[key] = str(value)` | 转普通字符串。 |
| L71 | `resolved_device` | 既记录用户请求的 device，也记录实际解析到 cpu/cuda。 |
| L72 | `selection_metric` | 明确记录 validation accuracy。 |
| L73 | 返回字典 | 可同时写 JSON 和 checkpoint。 |

### 11.5 L76–93：输出目录和防误覆盖

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L76 | 函数签名 | 接收具体 run 目录和覆盖授权。 |
| L77–83 | `generated_names` 元组 | 列出本程序拥有并可覆盖的五种产物，不递归删除整个目录。 |
| L84 | 列表推导式 | 对每个名字拼接路径，只收集实际存在的文件。Path 的 `/` 在这里是路径拼接。 |
| L85 | 有旧产物且未授权 | 进入保护分支。 |
| L86–89 | `FileExistsError` | 要求换 output dir 或显式 overwrite，防止不同实验混写。括号内相邻字符串自动拼接。 |
| L90 | `if overwrite:` | 只有显式授权才删除。 |
| L91–92 | 遍历 existing 并 unlink | 只删除已知生成文件，不删除未知用户文件。 |
| L93 | `mkdir(parents=True, exist_ok=True)` | 创建 run 目录；已有空目录也不报错。 |

### 11.6 L96–124：准备数据和配置

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L96 | `def main() -> None:` | 封装主流程，避免 import 时立即训练。 |
| L97 | 解析 args | 获取命令行配置。 |
| L98 | 验证 args | 在产生昂贵副作用前失败。 |
| L99 | 设置 seed | 必须早于模型初始化和 DataLoader 构造。 |
| L100 | 解析 device | 得到真实设备。 |
| L101 | `args.output_dir / args.model` | 例如 `runs/cifar10/cnn`。 |
| L102 | 准备目录 | 防覆盖并清理明确授权的旧产物。 |
| L104 | 生成 config | 包括实际 device 和选模指标。 |
| L105–108 | 打印首行配置 | 运行开始就暴露 device、模型、数据和 selection metric，便于发现跑错配置。 |
| L110–119 | 构造 train/validation | 逐项使用关键字参数，避免长位置参数错位；此处没有 test。 |
| L120–123 | 打印 split sizes | `len(loader.dataset)` 读取实际 quick/full 数量，并明确 `test=not_loaded`。 |
| L124 | 保存 config JSON | 数据成功构造后才写有效 run 配置；下载失败不会留下看似完成的 config。 |

### 11.7 L126–138：模型、loss、AdamW 和 cosine

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L126 | `build_model(...).to(device)` | 先构造随机模型，再把注册参数和 buffer 移到 device。链式调用返回模型自身。 |
| L127 | `CrossEntropyLoss()` | 输入 logits 和 int64 类别索引，默认 batch mean。 |
| L128–132 | `torch.optim.AdamW` | 把全部模型参数引用交给优化器，并设置初始学习率与解耦 weight decay。 |
| L133 | `scheduler = None` | 先建立统一变量；选择 none 时 checkpoint 仍可保存 None。 |
| L134 | 检查 cosine | 只有配置为 cosine 才构造。 |
| L135–138 | `CosineAnnealingLR` | 绑定 optimizer，`T_max=args.epochs` 表示一个余弦周期覆盖整个训练。 |

AdamW 的直观更新可写为：

$$
\theta_{t+1}
\approx
(1-\eta_t\lambda)\theta_t
-
\eta_t
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

第一项对参数做解耦衰减，第二项是 Adam 自适应梯度更新。它和把 L2 项简单混进 Adam 梯度
并非在所有情况下等价。

余弦学习率：

$$
\eta_t
=
\eta_{\min}
+
\frac{1}{2}
(\eta_{\max}-\eta_{\min})
\left(1+\cos\frac{\pi t}{T_{\max}}\right).
$$

前期步长大，后期逐渐减小，帮助在训练末期细化参数。

### 11.8 L140–187：epoch 主循环

#### 开始、计时和训练验证

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L140 | history 路径 | 指向当前 run 的 CSV。 |
| L141 | `best_val_accuracy = -1.0` | 合法准确率在 `[0,1]`，所以第一轮一定保存 best。 |
| L143 | `range(1, epochs+1)` | 右端不含，因此得到 1 到 epochs，日志符合人类编号。 |
| L144 | `time.perf_counter()` | 高分辨率单调计时器，适合墙钟耗时差。 |
| L145 | 读取 optimizer 当前 lr | `param_groups[0]` 是第一参数组；记录的是本轮实际使用的学习率。 |
| L146–148 | `train_one_epoch` | 完整遍历 train，多次更新参数。 |
| L149–151 | `evaluate(validation_loader)` | 用更新后的模型在 validation 评估，不更新参数。 |
| L152 | 耗时差 | 包含本轮 train + validation，不包含 checkpoint 写盘和最终绘图。 |

#### CSV 一行

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L154–162 | `row` 字典 | 统一记录 epoch、train/val loss/accuracy、lr 和耗时；插入顺序成为 CSV 列顺序。 |
| L163 | `append_history` | 立刻写盘，即使后面中断，已完成轮的历史仍保留。 |

#### best、scheduler 和 checkpoint

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L165 | `improved = current > best` | 必须在更新 best 前判断；严格大于表示相同 accuracy 保留更早 checkpoint。 |
| L166 | `max(old,current)` | 更新历史最佳值。 |
| L167 | scheduler 非空检查 | 选择 none 时跳过。 |
| L168 | `scheduler.step()` | 将 optimizer 学习率推进到下一 epoch；因此 CSV 的 lr 是本轮旧值，checkpoint scheduler state 是下一轮起点。 |
| L169–177 | 保存 `last.pt` | 每轮无条件保存当前模型、AdamW、scheduler、epoch、best 和 config。 |
| L178 | `if improved:` | 只有 validation accuracy 创造新高才进入。 |
| L179–187 | 保存 `best.pt` | 内容 schema 与 last 相同，但模型状态来自验证最佳轮。 |

为什么先 `scheduler.step()` 再保存？如果未来实现恢复训练，checkpoint 内 optimizer/scheduler
状态应该准备好从下一轮继续。当前工程已保存所需状态，但还没有 `--resume` 入口，不能把
“状态足够”写成“已经支持一键续训”。

### 11.9 L189–212：日志、曲线和入口保护

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L189–196 | 多段 f-string | `:02d` 补两位 epoch；`:.4f` 四位小数；`:.2%` 百分比；`:.6g` 六位有效数字；`:.1f` 一位耗时。 |
| L198 | `plot_history(...)` | 所有 epoch 结束后从 CSV 自动生成 `curves.png`，返回路径。 |
| L199–203 | 参数量生成器求和 | 遍历参数，只统计 `requires_grad=True`；`numel()` 返回元素数量。 |
| L204–208 | 最终打印 | 报告最佳 val accuracy、千位分隔参数量、产物目录和曲线路径。 |
| L211 | `if __name__ == "__main__":` | 以脚本/`-m` 执行时为真；被 import 时为假。Windows 多进程 DataLoader 也依赖入口保护。 |
| L212 | `main()` | 只有真正作为入口运行时启动训练。 |

## 12. `evaluate.py`：最终 test 的独立边界

### 12.1 L1–13：导入

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L1 | 模块 docstring | 强调 saved checkpoint、once、held-out test。 |
| L3 | future annotations | 同前。 |
| L5–6 | argparse、Path | 解析 checkpoint 路径和可选配置。 |
| L8 | `nn` | 构造与训练相同的交叉熵。 |
| L10 | `make_test_loader` | 只有本入口导入 test loader。 |
| L11 | engine 函数 | 共享评估口径和 device 解析。 |
| L12 | 模型工厂 | 按 checkpoint config 重建结构。 |
| L13 | checkpoint loader | 共享安全加载方式。 |

### 12.2 L16–25：评估参数

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L16–17 | 解析器 | description 出现在帮助文本。 |
| L18 | 位置参数 `checkpoint` | 没有 `--`，调用时必须给路径，并转 Path。 |
| L19 | 可选 dataset | 不提供时为 None，main 回退到 checkpoint 配置。 |
| L20 | data dir | 默认 `data`。 |
| L21 | batch 256 | 推理不保存反向图，通常可比训练 batch 大。 |
| L22 | worker | 默认 0。 |
| L23 | device | 默认 auto。 |
| L24 | test limit | quick evaluation；正式报告不要设置。 |
| L25 | 返回解析结果 | 得到 Namespace。 |

### 12.3 L28–59：独立评估主流程

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L28–29 | main 与 args | 启动独立评估。 |
| L30 | 解析 device | checkpoint 映射、模型和数据使用同一设备。 |
| L31 | 加载 checkpoint | 返回保存的状态字典。 |
| L32 | `checkpoint.get(...) != ...` | 用 `get` 避免旧文件缺键时直接 KeyError，并检查这次 run 的 best 选模协议。best 和 last 都保存该字段。 |
| L33 | ValueError | 拒绝选模协议不是 validation accuracy 的旧/异构 checkpoint，防止协议混淆；它不会根据文件名判断传入的是 best 还是 last。 |
| L35 | 取 config | 方括号索引缺键会 KeyError，表示 checkpoint schema 不兼容。 |
| L36 | model name | 必须用保存时模型结构。 |
| L37 | `args.dataset or config["dataset"]` | 命令行显式覆盖优先，否则使用保存值。 |
| L38–45 | 构造 test loader | 此时才加载 test；seed 对 synthetic 有用；旧 config 没 seed 时默认 42。 |
| L46 | 新建模型并移 device | 此时仍是随机参数，只是结构正确。 |
| L47 | `load_state_dict` | 按键和 shape 将模型参数/BN buffer 注入；结构不一致会报错。 |
| L48 | 共享 `evaluate` | 使用 test loader 与交叉熵，得到独立指标。 |
| L49–55 | 输出 | 报告模型、best epoch、选模指标、test loss/accuracy 和样本数。 |
| L58–59 | 入口保护 | 以模块运行时才执行 main。 |

为什么先 `build_model` 再 `load_state_dict`？state dict 保存状态，不保存完整 Python 类定义。
这种方式比 `torch.save(model)` 更清晰，也更适合代码升级。

因此正式命令仍应显式传 `best.pt`。当前检查能证明 run 的选模规则合规，但 `last.pt` 也
带相同 selection metric，所以代码不会替你识别文件角色。最终 test 之后若因为结果不好
又回去反复改超参数，再重新看同一 test，test 也会逐渐参与选择。严格做法是所有调参都看
validation，最终方案冻结后再做一次 test 报告。

## 13. `plot_history.py`：把 CSV 变成诊断证据

### 13.1 L1–12：导入和后端

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L1 | 模块 docstring | 输入 train/validation CSV，输出曲线。 |
| L3 | future annotations | 同前。 |
| L5 | argparse | 命令行入口。 |
| L6 | csv | 读取标准 CSV。 |
| L7 | Path | 输入输出路径。 |
| L9 | `import matplotlib` | 先导入顶层包，以便设置后端。 |
| L11 | `matplotlib.use("Agg")` | 选择无需图形界面的静态后端，服务器也可保存 PNG；必须早于 pyplot 导入。 |
| L12 | `pyplot as plt` | 导入绘图接口。位置不是随意的，要在设置 Agg 后。 |

### 13.2 L15–44：`plot_history`

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L15 | 签名 | 输入 CSV 和输出路径，返回最终 Path；可被 train 和 CLI 复用。 |
| L16–17 | 两个参数转 Path | 接受字符串或 Path。 |
| L18 | with 打开 CSV | UTF-8、newline 空字符串，退出自动关闭。 |
| L19 | `list(csv.DictReader(file))` | 用表头作键，一次读完 epoch 级小文件；所有值最初是字符串。 |
| L20–21 | 空历史检查 | 无行时不能画图，给出包含路径的错误。 |
| L23 | epoch 转 int | 列表推导式恢复数值类型。 |
| L24–27 | 四列转 float | 恢复 train/val loss/accuracy。 |
| L29 | `plt.subplots(1,2,figsize=(10,4))` | 创建一行两列子图，返回 Figure 与两个 Axes。尺寸单位英寸。 |
| L30–31 | 左图两条 loss | marker 标记每个 epoch；label 用于图例。 |
| L32 | `axes[0].set(...)` | 一次设置标题、横轴和纵轴。 |
| L33 | legend | 区分 train/validation。 |
| L35–36 | 右图两条 accuracy | 使用同一 epoch 横坐标。 |
| L37 | 右图设置 | accuracy 轴固定 `(0,1)`，不同实验更易公平比较。 |
| L38 | 右图 legend | 同前。 |
| L39 | `tight_layout()` | 自动调整子图间距，减少标签重叠。 |
| L41 | 创建输出父目录 | 支持嵌套路径。 |
| L42 | `savefig(..., dpi=160)` | 保存 PNG，dpi 控制栅格分辨率。 |
| L43 | `plt.close(figure)` | 主动释放 Figure，避免批量画图时内存累积。W2 版本没有这一行。 |
| L44 | 返回输出 Path | train 可打印或继续处理。 |

### 13.3 L47–61：绘图 CLI

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L47 | parse_args 签名 | 返回 Namespace。 |
| L48 | 创建解析器 | 帮助说明。 |
| L49 | 位置参数 history | 必须提供 CSV。 |
| L50 | 可选 output | 默认当前目录 `curves.png`。 |
| L51 | 返回解析结果 | 同前。 |
| L54–55 | main 与 args | 开始一次画图任务。 |
| L56 | 调可复用函数 | 取得实际输出路径。 |
| L57 | 打印路径 | 给用户明确反馈。 |
| L60–61 | 入口保护 | 以模块运行时执行。 |

## 14. `tests/test_smoke.py`：测试实验契约

W2 的测试主要问“模型和一轮训练能不能跑”。W3 进一步问：

- 划分是否可复现、互斥且完整；
- 随机增强是否只在 train；
- 真实 CIFAR 分支在断网时能否通过 mock 检查；
- CLI 是否真的生成全部产物；
- best 是否来自 validation 最大值；
- last 是否来自最后一轮；
- 独立 evaluate 是否遵守 validation-selected 协议。

### 14.1 L1–24：测试依赖逐行

| 行号 | 代码 | 作用 |
|---|---|---|
| L1 | `import csv` | 读取训练生成的 history。 |
| L2 | `import json` | 读取 config。 |
| L3 | `subprocess` | 启动真正的 `python -m ...` 子进程，测试 CLI 边界。 |
| L4 | `sys` | 使用当前解释器 `sys.executable`，避免子进程跑错环境。 |
| L5 | `tempfile` | 创建自动清理的临时输出目录。 |
| L6 | `unittest` | Python 标准库测试框架。 |
| L7 | `Path` | 仓库路径与产物路径。 |
| L8 | `patch` | 临时替换 `datasets.CIFAR10`，实现无网络测试。 |
| L10 | `numpy as np` | 构造假的 HWC uint8 图像。 |
| L11 | `torch` | Tensor、Dataset、optimizer、checkpoint。 |
| L12 | `PIL.Image` | 将 ndarray 变成与 torchvision CIFAR-10 一致的 PIL 输入。 |
| L13 | `nn` | 构造交叉熵。 |
| L14 | torchvision transforms | 检查 Compose 中是否存在随机增强类型。 |
| L16–22 | data imports | 引入 synthetic、transform、三种 loader/划分接口。括号允许多行导入。 |
| L23 | engine imports | 直接测试训练和评估函数。 |
| L24 | model factory | 构造 CNN。 |

### 14.2 L27–42：仓库根和两个基础测试

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L27 | `Path(__file__).resolve().parents[2]` | 从 `tests/test_smoke.py` 向上两级得到主仓根，供 subprocess 设置工作目录。`__file__` 是当前文件路径。 |
| L30 | 测试类继承 TestCase | 方法名以 `test_` 开头会被 discover 自动执行。 |
| L31 | shape 测试声明 | 测模型统一契约。 |
| L32 | 随机 `(4,3,32,32)` | batch 4，RGB 32×32。 |
| L33 | 构造、前向、shape 断言 | 转 tuple 后与 `(4,10)` 比较。 |
| L35 | 划分测试声明 | 同时检查 determinism、disjoint、coverage。 |
| L36–37 | 同参数调用两次 | 给相同 seed，得到 a/b 两份结果。 |
| L38 | 两次结果完全相等 | 验证确定性。 |
| L39–40 | 长度 80/20 | 验证切分数量。 |
| L41 | `set(train).isdisjoint(val)` | 验证没有索引同时属于两边。 |
| L42 | 集合并等于 `range(100)` | 验证无遗漏、无额外索引。`|` 是集合并。 |

### 14.3 L44–60：增强只属于 train

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L44 | 测试声明 | 检查 transform 结构，而不依赖随机抽样恰好发生。 |
| L45 | 解包两套 transform | 取得 train/evaluation Compose。 |
| L46–48 | `any(isinstance(...RandomCrop...))` | 遍历 train transforms，要求至少存在一个 RandomCrop。`any` 遇到 True 会短路。 |
| L49–54 | 水平翻转断言 | 同理要求 train 中有 RandomHorizontalFlip；多行生成器表达式提高可读性。 |
| L55–60 | `assertFalse(any(...))` | evaluation 中不能出现 crop 或 flip；`isinstance` 第二参数可以是类型元组。 |

### 14.4 L62–97：synthetic 单样本与单轮训练

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L62 | 测试声明 | 同时覆盖 Dataset 契约和 engine。 |
| L63 | 构造 23 个样本 | 非 10 倍数也应工作；固定 seed。 |
| L64 | 取第一个样本并解包 | 触发 `__getitem__`。 |
| L65 | image shape | 单样本无 batch 维。 |
| L66 | image dtype | 应与模型默认 float32 权重匹配。 |
| L67 | label dtype | 交叉熵要求 int64 类别索引。 |
| L69–76 | 构造小 loaders | 无需联网，train 40、val 20、batch 10。`data_dir` 在 synthetic 分支不使用。 |
| L77 | 固定模型初始化 seed | 提高测试稳定性。 |
| L78 | 构造 CNN | 覆盖模型工厂。 |
| L79 | 交叉熵 | 与正式训练一致。 |
| L80 | AdamW | 与正式训练同类优化器。 |
| L81 | 参数快照 | `detach()` 断开 autograd，`clone()` 复制数值；只 detach 会共享底层存储。 |
| L83–85 | CPU 训练一轮 | 执行真实 forward/backward/step。 |
| L86–88 | CPU validation | 执行 eval + inference mode。 |
| L89–94 | 至少一个参数变化 | `zip` 成对遍历旧/新参数，`allclose` 检查数值近似，`any` 要求至少一项变化。 |
| L95 | loss > 0 | 只检查数值合法，不要求达到某精度。 |
| L96–97 | accuracy 在 `[0,1]` | 防明显计数错误。 |

### 14.5 L99–142：用 mock 离线走真实 CIFAR 分支

#### Fake Dataset

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L99 | 测试声明 | 目标是覆盖 `dataset_name="cifar10"`，但不联网。 |
| L100 | 内部 `FakeCifar10(Dataset)` | 只在此测试使用，不污染项目代码。 |
| L101 | 与 torchvision 相同的构造签名 | 让生产函数无需知道自己被替换。未使用的 root/download 仍保留接口。 |
| L102 | train 100 / test 30 | 根据布尔 train 模拟不同 split 大小。条件表达式一行选择。 |
| L103 | 保存 transform | 生产代码传入的随机/确定性处理会真实执行。 |
| L105–106 | `__len__` | 返回 fake 大小。 |
| L108 | `__getitem__` | 定义单样本访问。 |
| L109 | HWC uint8 零图像 | 模拟 torchvision/PIL 常见输入格式。 |
| L110 | ndarray→PIL→transform | 真实执行 Compose，再返回 `index % 10` 标签。 |

#### 临时替换和断言

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L112 | `with patch("...datasets.CIFAR10", FakeCifar10):` | 在 with 块内把 data 模块所见的 CIFAR10 临时替换，退出后自动恢复。 |
| L113–121 | 构造 train/val | 走真实分支：100 样本先分 80/20，再 limit 到 30/10。 |
| L122–127 | 构造 test | fake official test 30，再 limit 到 12。 |
| L129–131 | 三个长度断言 | 验证 limit 与独立 test 构造。 |
| L132–136 | Subset 索引互斥 | 直接读取两个 Subset 的 indices，验证没有泄漏。 |
| L137–138 | 各取一个 batch | 真正触发 PIL transform 和 DataLoader collate。 |
| L139–140 | batch shape | train/validation 都应是 `(8,3,32,32)`。 |
| L141–142 | 标签 dtype | DataLoader 将 Python int 标签堆叠为 int64 Tensor。 |

### 14.6 L144–226：端到端 CLI 集成测试

这是最接近真实用户操作的一项测试。

#### 临时目录和训练命令

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L144 | 测试声明 | 验证完整 CLI 和产物语义。 |
| L145 | `TemporaryDirectory()` | 创建临时目录；with 结束后自动清理，不污染仓库 runs。 |
| L146 | output root | 在临时目录下拼 `runs`。 |
| L147–165 | command 列表 | 每个命令行 token 独立元素，避免 shell 引号问题；使用当前解释器、模块模式、synthetic、2 epochs、CPU 和小数据。 |
| L148 | `sys.executable` | 保证子进程与测试处于同一 Conda/Python 环境。 |
| L149–150 | `-m project2_cifar10.train` | 测试真实模块入口，不绕过 argparse。 |
| L164 | `str(output_root)` | subprocess 参数必须是字符串/路径兼容对象，这里显式转字符串。 |
| L166–173 | `subprocess.run` | 工作目录设主仓；捕获 stdout/stderr 为文本；`check=False` 让测试自己断言返回码；120 秒防卡死。 |
| L174 | return code 断言 | 训练失败时把 stderr 作为断言消息展示。 |

#### 检查产物、CSV 和 checkpoint

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L176 | `run_dir = output_root / "cnn"` | 对应 train.py 拼接模型名。 |
| L177–178 | 遍历五个文件名 | 每个都必须是普通文件；第二参数 `name` 作为失败提示。 |
| L179 | PNG size > 0 | 只存在空文件不算成功绘图。 |
| L181 | with 打开 history | UTF-8 CSV。 |
| L182 | DictReader 读全部行 | 两个 epoch 应有两行。 |
| L183 | 行数断言 | 验证每轮都记录。 |
| L184 | 必须有 val_accuracy | 选模证据进入日志。 |
| L185 | 不能有 test_accuracy | 训练历史不应混入 test。 |
| L187 | 读取 config JSON | `read_text` 后 `json.loads` 恢复字典。 |
| L188 | selection metric 断言 | 配置明确 validation accuracy。 |
| L189–190 | 加载 best/last | CPU 映射与受限反序列化。 |
| L191 | 从 CSV 算最大 val | 不相信 checkpoint 自己声称，独立从证据计算。 |
| L192–196 | 找首次最大 epoch | `next` 取得第一个满足最大准确率的 epoch，对应严格大于的 tie 语义。 |
| L197 | best 选模字段 | checkpoint 协议检查。 |
| L198 | best 历史值 | 与 CSV 最大值近似相等。 |
| L199 | best epoch | 必须是首次最大轮。 |
| L200 | last epoch = 2 | last 必须来自最后完成轮。 |
| L201 | last 记录同一历史 best | 即使 last 模型不是 best，它仍应知道历史最佳值。 |
| L202–203 | optimizer/scheduler 字段 | last 保存恢复训练所需核心状态。 |

#### 独立 evaluate 子进程

| 行号 | 代码 | 逐行解释 |
|---|---|---|
| L205–223 | 第二次 `subprocess.run` | 启动独立 evaluate，传刚生成的 best、synthetic test、30 样本、CPU；同样捕获输出并限时。 |
| L207 | 当前解释器 | 同前。 |
| L208–210 | 模块和 checkpoint | 真正走加载—重建—注入—评估链。 |
| L211–216 | 可选参数 | 显式 synthetic，限制 test，CPU。 |
| L218–222 | 运行设置 | 与训练子进程相同的稳定边界。 |
| L224 | return code | 评估必须成功。 |
| L225 | stdout 包含选模协议 | 证明独立入口识别 validation-selected。 |
| L226 | stdout 包含 test accuracy | 证明 test 流程实际结束。 |
| L229–230 | 测试入口保护 | 直接运行文件时启动 unittest；discover 也能发现测试类。 |

这些测试仍然没有证明：

- 真实 CIFAR-10 能达到某个 accuracy；
- 30 epochs 是最佳配置；
- 单 seed 结论稳定；
- checkpoint 已有一键 resume；
- 模型优于 ResNet。

Smoke test 是工程安全网，不是正式实验。

## 15. `requirements.txt` 与 `README.md`

### 15.1 四行依赖

| 行号 | 依赖 | 用途 |
|---|---|---|
| L1 | `torch` | Tensor、autograd、模型、loss、AdamW、scheduler、DataLoader、checkpoint。 |
| L2 | `torchvision` | CIFAR-10 与图像 transforms；其依赖会提供 Pillow。 |
| L3 | `numpy` | seed 与测试 fake 图像。 |
| L4 | `matplotlib` | 从 CSV 绘制曲线。 |

当前未固定版本，适合学习环境；正式复现必须记录实际版本：

```powershell
python -c "import torch, torchvision, numpy, matplotlib; print(torch.__version__, torchvision.__version__, numpy.__version__, matplotlib.__version__)"
```

### 15.2 README 的职责

README 不是可执行源码，不做机械逐句解释。它应回答：

- 项目为什么存在；
- 数据协议是什么；
- 怎样安装、测试、训练、评估和画图；
- 每个产物语义是什么；
- 哪些结果真实运行过；
- 哪些仍是待完成项；
- 实验有哪些限制。

当前 README 已明确区分 synthetic smoke 与真实 CIFAR-10 结果，避免把“管道能跑”写成
“模型在真实任务上有效”。

## 16. 把跨文件调用链完整走三遍

### 16.1 第一次：程序启动到第一个 batch

1. Python 执行 `python -m project2_cifar10.train`。
2. `train.py:L211–212` 的入口保护调用 `main()`。
3. `parse_args` 把命令行字符串转换为带类型的 Namespace。
4. `_validate_args` 检查 epochs、batch size、lr、weight decay、validation size。
5. `set_seed` 在模型初始化和 loader 构造前设置主要随机系统。
6. `resolve_device` 将 auto 解析为 cuda/cpu。
7. `_prepare_run_directory` 防止旧实验被静默覆盖。
8. `make_train_val_loaders` 进入 `data.py`：
   - 创建 train/evaluation transforms；
   - 构造两份 `train=True` CIFAR Dataset；
   - 生成一次随机排列；
   - 切为 train/validation 索引；
   - 分别用 Subset 包装；
   - 构造两个 DataLoader；
   - 整个过程没有构造 `train=False` test。
9. `build_model("cnn")` 构造随机初始化的 BasicCNN。
10. `model.to(device)` 移动参数和 BatchNorm buffer。
11. 构造 CrossEntropyLoss、AdamW 和 cosine scheduler。
12. epoch 开始，`train_one_epoch` 调用 `model.train()`。
13. train loader 打乱索引，Subset 找到 augmented Dataset，再调用 transform。
14. DataLoader 将单样本堆叠为 images `(B,3,32,32)` 和 labels `(B,)`。
15. engine 把数据移到 device，执行训练五步。

### 16.2 第二次：一个 epoch 结束

1. `train_one_epoch` 返回按样本加权的 train loss/accuracy。
2. `evaluate(validation_loader)`：
   - `model.eval()`；
   - `inference_mode()`；
   - 确定性 validation transform；
   - 返回同口径 val 指标。
3. 计时器得到 train + validation 耗时。
4. `append_history` 立即写 CSV。
5. 当前 val accuracy 与历史 best 比较。
6. scheduler 推进到下一轮学习率。
7. 当前状态总是保存为 last。
8. 只有严格提高时保存 best。
9. 当前 epoch 的格式化日志打印到终端。

### 16.3 第三次：训练结束后的 test

1. train 调 `plot_history`，从 CSV 生成曲线。
2. 用户另外执行 `project2_cifar10.evaluate best.pt`。
3. `load_checkpoint` 将 Tensor 映射到目标 device。
4. evaluate 检查 selection metric 是 validation accuracy。
5. `make_test_loader` 此时才构造 `datasets.CIFAR10(train=False)`。
6. `build_model` 重新构造空结构。
7. `load_state_dict` 注入模型权重和 BatchNorm buffer。
8. 共享 `engine.evaluate` 遍历 test。
9. 打印一次 test 结果，不改 best/last，不写入训练 history。

若你能脱离讲义口头走完这三遍，并指出每一步对应文件，才算建立工程全局理解。

## 17. 理论总复盘：结构、优化与泛化如何连接

### 17.1 CNN 的三种归纳偏置

**局部连接**：一个卷积单元先观察小邻域，符合边缘、纹理由相邻像素组成的结构。

**权重共享**：同一卷积核在所有空间位置复用。某种边缘出现在左上或右下，都可以被同一
核检测。

**平移等变**：输入平移时，中间特征响应大致相应平移。它不是严格的最终平移不变；
池化、global average pooling 和数据增强共同提高分类对小平移的鲁棒性。

归纳偏置（inductive bias）是模型在看数据前由结构获得的偏好，不是“CNN 必然正确”的
保证。

### 17.2 Pooling 和 global average pooling

MaxPool 在局部窗口取最大响应：

$$
Y_{c,i,j}
=
\max_{(u,v)\in\Omega_{i,j}} X_{c,u,v}.
$$

它降低空间分辨率、减少后续计算，并让后层相对原图的感受野扩大。

Global average pooling 对每个通道：

$$
g_c
=
\frac{1}{HW}
\sum_{i=1}^{H}
\sum_{j=1}^{W}
X_{c,i,j}.
$$

本模型把 `(B,128,8,8)` 变为 `(B,128,1,1)`。与把 8192 个数直接接巨大 Linear 相比，
分类头参数更少，也迫使每个通道表达“某种特征在全图的总体存在程度”。

### 17.3 训练 accuracy 为什么可能暂时低于 validation

训练指标是在以下条件累计的：

- Dropout 开启；
- 数据有随机 crop/flip；
- 参数在一个 epoch 内不断更新；
- 前几个 batch 使用较旧的参数。

validation 则在 epoch 结束后：

- 使用最终参数；
- Dropout 关闭；
- 输入无随机增强；
- BatchNorm 使用 running stats。

所以前期 val accuracy 高于 train 并不必然表示数据泄漏。要结合数据划分、模式和曲线判断。

### 17.4 训练增强与正则化的关系

训练增强扩大了模型实际看到的输入分布。粗略写：

$$
\min_\theta
\mathbb E_{(x,y)\sim D_{train}}
\mathbb E_{T\sim\mathcal T}
\left[
L(f_\theta(T(x)),y)
\right].
$$

模型不是只拟合原始 `x`，还要对合法变换 `T(x)` 保持分类稳定。这是一种数据层面的
正则化。

Weight decay 是参数层面的约束；Dropout 是表示层面的随机扰动；它们目标都与减少过拟合
有关，但机制不同，不能简单看成同一个开关。

### 17.5 best accuracy 与 best loss 不是同一件事

本工程按 validation accuracy 选择 best。accuracy 只关心最大 logit 的类别是否正确；
cross entropy 还关心置信度。

例如两个模型都把同样数量样本分对，但一个对错误样本极度自信，它的 loss 会更大。因此：

- best accuracy 适合直接对齐分类指标；
- best loss 可能对概率质量更敏感；
- 选择哪个必须在实验前规定，不能跑完后挑对自己最有利的。

当前 checkpoint 明确写 `selection_metric=validation_accuracy`，避免事后含糊。

## 18. 怎样阅读训练曲线

不要只看最后一个点。至少同时观察：

| 现象 | 可能原因 | 优先验证 |
|---|---|---|
| train/val loss 同降 | 正常学习 | 继续观察 gap 和平台 |
| train loss 降、val loss 升 | 过拟合 | 增强、weight decay、训练轮数 |
| train/val 都几乎不变 | 优化或数据错误 | 梯度、标签、lr、模型输出 |
| train 高、val 很低 | 过拟合或泄漏/分布问题 | 划分、transform、模式 |
| val 波动很大 | val 太小或仍有随机性 | validation size、随机增强、eval |
| accuracy 不动但 loss 下降 | 置信度改善但类别未改变 | 查看 logits/loss，不急着判失败 |
| 恢复后 lr/曲线突变 | 状态恢复不完整 | optimizer/scheduler/epoch |
| 一开始 NaN | 数值或输入错误 | finite、lr、loss 用法、除零 |

一次只改变一个变量。若同时换模型、学习率、增强和 batch size，就无法判断结果由谁造成。

### 18.1 一个基本过拟合判断

泛化间隙可粗略写：

$$
\operatorname{gap}
=
\operatorname{Acc}_{train}
-
\operatorname{Acc}_{val}.
$$

gap 大不自动等于 bug，但提示模型对训练分布拟合得远好于验证分布。要进一步看：

- train accuracy 是否在增强和 Dropout 开启状态统计；
- validation 是否足够大；
- 两边预处理是否除随机增强外保持一致；
- 是否存在重复样本或错误标签；
- val loss 是否已经上升。

## 19. 实际运行顺序与命令

所有命令从主仓根目录执行：

```powershell
cd C:\Users\qiuyinxi\Desktop\AI\2_成果作品\A-thousand-journey-s-beginning
conda activate ai-learn
```

### 19.1 先跑离线测试

```powershell
python -m unittest discover -s project2_cifar10/tests -v
```

当前实际结果：6/6 passed。它不下载 CIFAR-10。

### 19.2 再跑 synthetic 端到端

```powershell
python -m project2_cifar10.train `
  --dataset synthetic `
  --epochs 2 `
  --batch-size 32 `
  --limit-train 128 `
  --limit-val 64 `
  --device cpu `
  --output-dir runs/cifar10_smoke
```

观察第一段输出必须包含：

```text
selection_metric=validation_accuracy
test=not_loaded
```

再独立评估：

```powershell
python -m project2_cifar10.evaluate `
  runs/cifar10_smoke/cnn/best.pt `
  --dataset synthetic `
  --limit-test 64 `
  --device cpu
```

synthetic 数字只验证管道，不写进真实 CIFAR-10 成绩表。

### 19.3 正式 CIFAR-10 baseline

```powershell
python -m project2_cifar10.train `
  --dataset cifar10 `
  --epochs 30 `
  --batch-size 128 `
  --learning-rate 0.001 `
  --weight-decay 0.0005 `
  --validation-size 5000 `
  --num-workers 0 `
  --device auto `
  --seed 42 `
  --output-dir runs/cifar10
```

第一次会下载官方数据。若目录已有实验，优先换新的 output dir；只有明确需要删除本程序
生成的同名产物时才加 `--overwrite`。

训练完成后：

```powershell
python -m project2_cifar10.evaluate `
  runs/cifar10/cnn/best.pt `
  --batch-size 256 `
  --num-workers 0 `
  --device auto
```

训练自动生成曲线，也可手动重绘：

```powershell
python -m project2_cifar10.plot_history `
  runs/cifar10/cnn/history.csv `
  --output runs/cifar10/cnn/curves_manual.png
```

### 19.4 当前真实验证边界

2026-08-10 已验证：

- Python 3.11.15；
- PyTorch 2.11.0+cu130；
- torchvision 0.26.0+cu130；
- CUDA 能识别 RTX 5060 Laptop GPU；
- 6 项离线测试通过；
- synthetic train/evaluate/plot 完整通过；
- best/last/CSV/config/curve 语义通过集成测试。

曾尝试首次下载官方 CIFAR-10 做受限真实数据 smoke，但官方源在 5 分钟内未下载完成，因此
没有产生真实 CIFAR-10 指标；残缺下载和失败 run 已清理。后续正式结果必须由上一节全量
命令实际运行后填写。

## 20. 建议的第一个独立改动

为了把 AI 生成的参考工程转化为自己的学习成果，完成一个单变量工程改动：

> 给 BasicCNN 增加可配置的 `--dropout`，默认仍为 0.3。

你需要自己判断会影响哪些文件：

1. `models.py`：`BasicCNN(dropout=...)`；
2. `build_model`：接受并传递 dropout；
3. `train.py`：argparse 增加参数并用它构造模型；
4. `config.json`：因为 `vars(args)` 自动包含，确认实际写入；
5. checkpoint：config 自动包含，但 evaluate 必须按 config 重建；
6. `evaluate.py`：从 config 恢复 dropout；
7. tests：检查不同 dropout 值的模型仍输出 `(B,10)`，并检查 config。

这个任务的关键不是改一行 `p=0.3`，而是理解“结构超参数必须沿 train → config →
checkpoint → evaluate 整条链传播”。

## 21. 常见错误：按调用链定位

### 21.1 `No module named torch`

```powershell
python --version
python -c "import sys, torch; print(sys.executable); print(torch.__version__)"
```

若不是 `ai-learn`，先激活正确环境，不要立刻重装。

### 21.2 CIFAR-10 下载很慢或中断

- 先确认 `data/cifar-10-batches-py` 是否完整；
- 残缺 `cifar-10-python.tar.gz` 可能无法通过完整性检查；
- 下载问题与模型代码错误分开诊断；
- 离线开发先跑 6 项测试，正式数据就绪后再跑真实分支。

不要把“下载失败”误诊为 CNN 或 DataLoader 逻辑失败。

### 21.3 `Expected target ... Long`

检查：

```text
labels shape: (B,)
labels dtype: torch.int64
```

不要把标签转 float，也不要在当前接口手工 one-hot。

### 21.4 device 不一致

典型错误：

```text
Expected all tensors to be on the same device
```

依次打印模型第一个参数、images、labels 的 device。新建 mask/常量也要放到正确 device。

### 21.5 Linear 或卷积 shape 错误

逐层打印：

```python
print(images.shape)
features = model.features(images)
print(features.shape)
```

当前契约必须是：

```text
(B,3,32,32) → (B,128,1,1) → (B,10)
```

不要看到错误就乱改 Linear 的 `in_features`；先用卷积输出公式和实际 shape 定位。

### 21.6 loss 为 NaN

按顺序检查：

1. `torch.isfinite(images).all()`；
2. logits 是否 finite；
3. 是否手动 softmax 后又传 CrossEntropyLoss；
4. 学习率是否过大；
5. 梯度何时第一次非有限；
6. 指标中是否出现除零。

### 21.7 validation 每次结果不同很多

优先检查：

- validation 是否意外用了 train transform；
- 是否漏掉 `model.eval()`；
- validation size 是否被 limit 得太小；
- BatchNorm running stats 是否可靠；
- 是否在 validation 中更新了模型。

### 21.8 best 和 last 混淆

- last：最后完成 epoch，适合继续训练状态；
- best：validation accuracy 最佳 epoch，适合最终 test。

不要默认 last 一定最好，也不要把 test accuracy 用来重写 best。

### 21.9 checkpoint 无法加载

检查：

- 当前 `models.py` 是否与保存时结构一致；
- `config["model"]` 是否匹配；
- 新增 dropout 不影响 state shape，但新增通道数会影响；
- 是否从 config 恢复所有结构超参数；
- `selection_metric` 是否存在且合规。

### 21.10 训练后没有曲线

检查 `history.csv` 是否存在且至少一行，列名必须包含：

```text
epoch,train_loss,train_accuracy,val_loss,val_accuracy,learning_rate,epoch_seconds
```

然后单独运行 plot CLI，把训练问题与绘图问题分离。

## 22. 读完后的闭卷自测

### 22.1 工程全局

1. 从 `python -m project2_cifar10.train` 开始，口头走完控制流。
2. 为什么 train.py 不导入 `make_test_loader`？
3. 每个文件的单一职责是什么？
4. 哪些对象会随 epoch 改变：模型、optimizer、scheduler、BatchNorm、CSV、best、last？
5. synthetic 通过为什么不能证明真实 CIFAR-10 accuracy？

### 22.2 数据与实验协议

6. 为什么需要两份 `datasets.CIFAR10(train=True)` 对象？
7. 怎样证明 train/validation 无交集且无遗漏？
8. validation 为什么不能有 RandomCrop/Flip？
9. test 为什么不能选择 best？
10. `limit_train` 为什么必须先验证正数？
11. `pin_memory=True` 与 `non_blocking=True` 什么时候可能有用？

### 22.3 CNN 理论

12. 写出 `Conv2d(3,32,3,padding=1,bias=False)` 的权重 shape、参数量和输出 shape。
13. 为什么连续两个 `3×3` 卷积感受野是 `5×5`？
14. 写出 BasicCNN 完整 shape 流。
15. 为什么当前卷积关闭 bias？
16. BatchNorm 有哪些可训练参数和哪些 buffer？
17. global average pooling 怎样把 `(B,128,8,8)` 变成 `(B,128,1,1)`？
18. 为什么模型最后不加 softmax？
19. 当前模型 288,746 个参数怎样逐项得到？

### 22.4 训练与状态

20. 训练五步的正确顺序是什么？顺序错了分别会怎样？
21. `zero_grad(set_to_none=True)` 与普通填零有什么语义差异？
22. `model.train()/eval()` 与 autograd 开关为什么不是同一件事？
23. validation loss 为什么按样本数加权？
24. CSV 记录的 learning rate 对应当前轮还是下一轮？
25. scheduler 为什么在保存 checkpoint 前 step？
26. best accuracy 持平时为什么保留更早轮？
27. checkpoint 里为什么同时保存 optimizer 和 scheduler state？
28. `--resume` 为什么同时恢复全局 RNG 和 DataLoader generator？
29. 临时文件 + replace 比直接写正式 checkpoint 好在哪里？

### 22.5 测试和诊断

30. mock CIFAR-10 测试解决了什么网络依赖？
31. 为什么 fake 图像要转 PIL，而不是直接把 HWC ndarray 给 RandomCrop？
32. `detach().clone()` 中缺少 clone 会怎样？
33. CLI 集成测试如何证明 best 来自 CSV 的 validation 最大值？
34. 为什么测试要求 history 不含 `test_accuracy`？
35. train accuracy 上升、val loss 上升时，你会先验证哪三件事？

能够不看代码回答 1、2、6、9、12、14、18、20、22、24、27、33，并能回源码指出对应
函数，才算真正理解这套工程。

## 23. 最终验收清单

- [ ] 能画出训练、验证和最终 test 的三段数据协议。
- [ ] 知道 9 个 Python 文件和 README/requirements 的职责。
- [ ] 逐文件读完所有有语义代码行，能说出输入、输出、副作用和不变量。
- [ ] 能写出 BasicCNN 的完整 shape 流和 288,746 参数量。
- [ ] 能解释卷积、池化、感受野、BatchNorm、Dropout、global average pooling。
- [ ] 能解释增强、标准化、AdamW、weight decay、cosine scheduler。
- [ ] 能解释 train/eval、autograd、交叉熵和样本加权指标。
- [ ] 能解释为什么 validation 选 best、test 只做最终评估。
- [ ] `ai-learn` 中 6 个离线测试全部通过。
- [x] synthetic 端到端跑通，但没有把它写成真实成绩。
- [ ] 能检查 config、history、best、last 和 curves 的契约。
- [ ] 独立加载 best checkpoint 并完成 test 入口。
- [ ] 自己完成一次 `--dropout` 跨文件修改并补测试。
- [x] 正式 CIFAR-10 全量训练完成，README 已记录真实命令、环境、曲线和 test 结果。
- [ ] 正式结果至少补一项单变量对照，且不根据 test 反复调参。

完成这些之后，这个项目才从“AI 给出的可运行参考工程”变成你能够解释、修改、验证并在
面试或导师交流中诚实讲清楚的个人学习成果。

## 24. 2026-08-16 工程更新：增强配置与真正的断点恢复

前 23 节完整解释了基础工程。源码新增功能后，旧节中的函数名和原理仍成立，但 `data.py`、
`utils.py`、`train.py` 的物理行号会后移。本节按当前源码补齐新增语义；阅读时以函数名和
当前内容定位。

### 24.1 `data.py`：三档增强

L13 新增：

```python
AUGMENTATION_CHOICES = ("none", "basic", "strong")
```

`build_transforms(augmentation="basic")` 先校验字符串，再按档位组装列表：

| 档位 | 随机步骤 | 固定步骤 |
|---|---|---|
| none | 无 | ToTensor + Normalize |
| basic | RandomCrop + RandomHorizontalFlip | ToTensor + Normalize |
| strong | basic + AutoAugment + RandomErasing | ToTensor + Normalize |

AutoAugment 必须在 ToTensor 前处理图像；RandomErasing 需要 Tensor，因此放在 ToTensor/
Normalize 后。validation/test 无论训练档位是什么，都调用 deterministic evaluation transform。

loader 新增 `augmentation` 和 `pin_memory` 参数。train 根据 resolved device 传
`pin_memory=device.type=="cuda"`，而不是仅凭机器“存在 CUDA”就让显式 CPU 实验锁页。

测试同时检查：none 没有 Crop/Flip，basic 有 Crop/Flip，strong 还有 AutoAugment/Erasing，
evaluation 三者都没有随机增强。

### 24.2 `utils.py`：seed 不等于完整恢复

`set_seed(seed, deterministic)` 除 Python/NumPy/PyTorch/CUDA seed 外，还设置 cuDNN benchmark/
deterministic 和 `torch.use_deterministic_algorithms(..., warn_only=True)`。

只在程序启动重新设 seed 不能恢复中断点：假设中断前已消耗一万次随机数，重新设初始 seed
会从第一个随机数重放。`capture_rng_state` 因此保存：

- Python `random.getstate()`；
- NumPy bit generator 名字、uint32 state、position 和 Gaussian cache；
- PyTorch CPU RNG tensor；
- 所有 CUDA device RNG tensors。

NumPy ndarray 转 list，是为了 checkpoint 能由 `torch.load(weights_only=True)` 加载简单容器；
恢复时再转回 `np.uint32` array。

checkpoint 还保存 `train_loader.generator.get_state()`。它控制 shuffle permutation 和 worker
seed；全局 PyTorch RNG 与 loader generator 是两个不同状态，缺一都不能在相同环境尽量接上。

### 24.3 `train.py` 新增参数

当前 `parse_args` L47–74 新增：

- `--augmentation {none,basic,strong}`；
- `--deterministic`；
- `--resume CHECKPOINT`。

`--resume` 与 `--overwrite` 互斥：前者保留并追加已有 run，后者删除明确生成物开始新 run。
恢复时 run_dir 直接使用 checkpoint 父目录，减少配置写到错误目录的风险。

### 24.4 resume L175–202 逐行逻辑

1. `load_checkpoint(..., map_location=device)` 允许 GPU checkpoint 在 CPU 加载；
2. 比较 model/dataset/augmentation/seed/scheduler，以及 batch、learning rate、weight decay、
   validation size、workers、limits、deterministic，拒绝训练身份变化；
3. load model state，包括 BN 参数和 buffers；
4. load optimizer state，包括 AdamW moments；
5. load scheduler state；
6. 要求 CSV 最后 epoch 等于 checkpoint epoch；
7. `start_epoch=checkpoint_epoch+1`；
8. 恢复 best validation；
9. 恢复 loader generator；
10. 最后恢复全局 RNG；
11. 要求总 `--epochs` 至少包含一个尚未训练的 epoch。

为什么最后恢复 RNG？构造 Dataset、model、optimizer 等步骤可能消耗随机数。先完成所有对象
构造/权重注入，再把 RNG 指针放回 checkpoint，下一轮才从正确位置继续。

恢复测试先跑 2 epochs，再从 last 恢复到总 epoch 3，断言 CSV 为 `[1,2,3]`，last epoch 为
3，并检查 checkpoint 含 RNG/loader state。它验证的是 epoch 边界恢复；进程在某个 epoch
中间崩溃时，本实现仍从该 epoch 开头重跑，不保存 batch 内位置。

### 24.5 CPU/GPU“一致”的准确含义

本工程的统一性包括：

- 同一模型、loss、engine 和指标定义；
- 所有 batch/模型通过明确 device 移动；
- checkpoint 使用 map_location；
- CUDA 不可用却显式请求 CUDA 时清楚报错；
- deterministic 模式尽量选择确定性 kernel。

它不承诺 CPU 与 GPU 逐 bit 相等。浮点运算顺序、并行 kernel、CUDA/cuDNN/库版本可能产生
微小差异，并在长训练中放大。可复现报告必须记录环境和硬件，不能把 seed=42 写成绝对保证。

### 24.6 真实 baseline 结果

环境：Windows、Python 3.11.15、PyTorch 2.11.0+cu130、torchvision 0.26.0+cu130、
NumPy 2.4.4、Matplotlib 3.10.8、RTX 5060 Laptop GPU。

固定配置：45,000 train / 5,000 validation、basic augmentation、AdamW lr 0.001、weight decay
0.0005、cosine、30 epochs、batch 128、seed 42、deterministic。

| 结果 | 数值 |
|---|---:|
| 参数量 | 288,746 |
| 最佳 validation | 88.42% |
| 最佳 epoch | 29 |
| epoch 30 validation | 88.20% |
| epoch 29 official test loss | 0.3982 |
| epoch 29 official test accuracy | 87.04% |
| test 样本 | 10,000 |

best 与 last 在真实运行中确实不同，说明分离两者不是形式主义。test 只评估 validation 选出的
epoch 29 一次；87.04% 超过计划 80% 门槛，但仍只是一个 seed 的基础 CNN 结果。

torchvision 读取 CIFAR 时在 NumPy 2.4 下给出 `VisibleDeprecationWarning`。训练/指标正常，
README 如实记录该警告；“命令 exit code 0”不等于“控制台绝对没有警告”。

### 24.7 更新后的闭卷问题

1. none/basic/strong 的 transform 顺序为什么不同？
2. 为什么 RandomErasing 放在 ToTensor 后？
3. 重新 set seed 为什么不是 resume？
4. 全局 RNG 与 DataLoader generator 分别控制什么？
5. resume 为什么先构造模型、最后恢复 RNG？
6. epoch 中间崩溃时当前工程能从哪个粒度恢复？
7. deterministic 为什么仍不能保证跨 CPU/GPU 逐 bit 相等？
8. 真实 best/last 分别是哪一轮，为什么最终 test 选 epoch 29？
