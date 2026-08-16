# Project 2：CIFAR-10 基础 CNN 训练工程

> 当前状态（2026-08-16）：工程、断点恢复测试和 30 轮真实 CIFAR-10 GPU baseline
> 均已完成。validation 选出的第 29 轮 checkpoint 在官方 test 上达到 **87.04%**。

这个项目把 `project1_mnist` 的 `data / models / engine / train / utils / tests`
模块风格延续到 CIFAR-10，并修正了 MNIST 入门工程用 test 集选 checkpoint 的实验协议：

- 官方 CIFAR-10 train split 固定划为 **45,000 train + 5,000 validation**；
- 训练期间完全不构造官方 test split；
- `best.pt` 只由 validation accuracy 选择；
- `last.pt` 每个 epoch 覆盖保存，可恢复 model、optimizer、scheduler 与随机状态；
- 最终 test 只通过独立 `evaluate` 入口执行。

## 1. 数据协议

```text
CIFAR-10 official train (50,000)
├── train      45,000：由 --augmentation 配置随机增强，随后归一化
└── validation  5,000：仅归一化

CIFAR-10 official test (10,000)
└── 训练结束后由 evaluate.py 独立加载：仅归一化
```

划分由 `--seed` 控制，默认 `42`，train/validation 索引互斥且覆盖官方训练集。
使用两份指向同一官方训练数据的 dataset 对象，是为了让 train subset 使用随机增强，
validation subset 使用确定性预处理。归一化常量为 CIFAR-10 训练集常用的逐通道统计量：

```text
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)
```

`--limit-train` 和 `--limit-val` 只用于调试；正式实验不要设置它们。

训练增强有三个可复现配置档位，validation/test 始终只做 `ToTensor + Normalize`：

| `--augmentation` | 训练预处理 | 用途 |
|---|---|---|
| `none` | 只归一化 | 无增强消融 |
| `basic`（默认） | RandomCrop + RandomHorizontalFlip | 本项目 baseline |
| `strong` | basic + CIFAR AutoAugment + RandomErasing | 更强增强实验 |

## 2. 模型与训练配置

`BasicCNN` 接收 `3×32×32` 输入，包含三组逐步扩通道的卷积特征提取、
BatchNorm、ReLU、两次最大池化和全局平均池化，最后使用 dropout 与线性分类头输出
10 类 logits。模型共有 **288,746 个可训练参数**。

默认训练配置：

| 项目 | 默认值 |
|---|---:|
| Optimizer | AdamW |
| Learning rate | `1e-3` |
| Weight decay | `5e-4` |
| Scheduler | cosine annealing |
| Epochs | 30 |
| Batch size | 128 |
| Loss | cross entropy |
| Validation size | 5,000 |
| Seed | 42 |
| Device | `auto`（CUDA 可用时使用 CUDA） |
| Augmentation | `basic` |

每轮 CSV 记录 `train_loss`、`train_accuracy`、`val_loss`、`val_accuracy`、
`learning_rate` 和 `epoch_seconds`。训练结束自动生成 `curves.png`。

## 3. 环境与测试

在仓库根目录 `A-thousand-journey-s-beginning` 执行。推荐使用已经配置好的
`ai-learn` Conda 环境。

安装依赖（环境尚未配置时）：

```powershell
conda activate ai-learn
python -m pip install -r project2_cifar10/requirements.txt
```

运行无需联网、不会下载 CIFAR-10 的 smoke tests：

```powershell
conda run --no-capture-output -n ai-learn python -m unittest discover `
  -s project2_cifar10/tests -v
```

这组测试覆盖：

1. CNN 输入/输出 shape；
2. train/validation 划分可复现、无交集且无遗漏；
3. `none/basic/strong` 三档增强内容正确，随机增强只出现在训练预处理；
4. mock 官方数据的 CIFAR-10 分支无需联网即可完成划分、预处理和 loader 构造；
5. synthetic 数据上的前向、反向、参数更新与 validation；
6. 完整训练 CLI 生成 CSV、曲线、best/last checkpoint，随后从 last 恢复并把历史从
   `[1, 2]` 连续写到 `[1, 2, 3]`，再独立加载 best 做 test。

## 4. 离线端到端 smoke run

synthetic 数据只是带人工图案的本地张量，用来验证工程管道，不代表 CIFAR-10 难度或
泛化性能。

```powershell
conda run --no-capture-output -n ai-learn python -m project2_cifar10.train `
  --dataset synthetic `
  --epochs 2 `
  --batch-size 32 `
  --limit-train 128 `
  --limit-val 64 `
  --device cpu `
  --output-dir runs/cifar10_smoke
```

独立加载 validation 选出的 best checkpoint，在 synthetic test 上走一遍评估入口：

```powershell
conda run --no-capture-output -n ai-learn python -m project2_cifar10.evaluate `
  runs/cifar10_smoke/cnn/best.pt `
  --dataset synthetic `
  --limit-test 64 `
  --device cpu
```

曲线在训练结束时已自动生成，也可以从 CSV 手动重绘：

```powershell
conda run --no-capture-output -n ai-learn python -m project2_cifar10.plot_history `
  runs/cifar10_smoke/cnn/history.csv `
  --output runs/cifar10_smoke/cnn/curves_manual.png
```

## 5. 正式 CIFAR-10 训练与最终评估

首次运行会由 torchvision 下载 CIFAR-10。下面是一条完整 baseline 命令：

```powershell
conda run --no-capture-output -n ai-learn python -m project2_cifar10.train `
  --dataset cifar10 `
  --epochs 30 `
  --batch-size 128 `
  --learning-rate 0.001 `
  --weight-decay 0.0005 `
  --validation-size 5000 `
  --num-workers 0 `
  --augmentation basic `
  --device auto `
  --seed 42 `
  --deterministic `
  --output-dir runs/cifar10_baseline_seed42
```

训练结束后才加载官方 test split，并且应优先评估 validation 选出的 `best.pt`：

```powershell
conda run --no-capture-output -n ai-learn python -m project2_cifar10.evaluate `
  runs/cifar10_baseline_seed42/cnn/best.pt `
  --batch-size 256 `
  --num-workers 0 `
  --device auto
```

若输出目录已有实验，程序默认拒绝覆盖。建议为新实验换 `--output-dir`；明确需要重跑同一
目录时才添加 `--overwrite`。

若训练被中断，`--epochs` 表示最终总轮数，用同一配置从 `last.pt` 恢复：

```powershell
conda run --no-capture-output -n ai-learn python -m project2_cifar10.train `
  --dataset cifar10 --epochs 30 --batch-size 128 `
  --learning-rate 0.001 --weight-decay 0.0005 `
  --augmentation basic --device auto --seed 42 --deterministic `
  --resume runs/cifar10_baseline_seed42/cnn/last.pt
```

恢复时会拒绝 model、dataset、augmentation、seed、scheduler、batch、optimizer、validation、
workers、limits 或 deterministic 与原实验不一致的配置，
并检查 CSV 最后一轮必须与 checkpoint 的 epoch 一致。checkpoint 保存了 Python、NumPy、
PyTorch CPU/CUDA RNG 和 DataLoader generator 状态。相同硬件与软件栈可最大限度复现续训；
不同 CPU/GPU、CUDA 或库版本仍不承诺逐 bit 相同。

## 6. checkpoint 与日志语义

```text
runs/cifar10/cnn/
├── config.json   # 解析后的命令行配置、实际 device、选模指标
├── history.csv   # 仅 train/validation 指标，不含 test 指标
├── curves.png    # 自动生成的 loss/accuracy 曲线
├── best.pt       # validation accuracy 首次严格提高时更新
└── last.pt       # 每个 epoch 更新，始终对应最后完成的 epoch
```

两个 checkpoint 均包含 model、optimizer、scheduler、epoch、历史最佳 validation
accuracy、配置、全局 RNG 和 DataLoader generator 状态。`selection_metric=validation_accuracy` 同时写入配置和 checkpoint；
独立评估入口会检查这个字段。若 validation accuracy 持平，保留最早达到该值的 best。

## 7. 2026-08-16 实际验证结果

验证环境：Windows、Python 3.11.15、PyTorch 2.11.0+cu130、torchvision
0.26.0+cu130、NumPy 2.4.4、Matplotlib 3.10.8。机器可用 RTX 5060 Laptop GPU，
离线测试明确使用 CPU，正式 baseline 使用 RTX 5060 Laptop GPU。

- `compileall`：通过；
- 离线单元/集成测试：**6/6 passed**；
- 仓库级 smoke train：成功完成 2 epochs，train 128 / validation 64，且输出明确显示
  `test=not_loaded`；
- 产物：`config.json`、`history.csv`、`curves.png`、`best.pt`、`last.pt` 全部生成；
- checkpoint：`best.pt` 与 `last.pt` 均可读取，并包含恢复所需状态；
- 断点恢复：离线训练先完成 2 轮，再从 `last.pt` 恢复到第 3 轮，CSV epoch 为
  `[1, 2, 3]`，无重复或缺口；
- 独立 synthetic test：成功加载 epoch 2 的 best，输出
  `selected_by=validation_accuracy`，64 个样本流程完整结束；
- 手动重绘：`curves_manual.png` 成功生成。
- 正式 baseline：官方 45,000 train / 5,000 validation，30 epochs，`basic` 增强，
  AdamW，cosine scheduler，seed 42，deterministic 模式；
- 最佳 validation：**88.42%（epoch 29）**；epoch 30 validation 为 88.20%；
- 独立官方 test：只评估 validation 选出的 epoch 29 `best.pt`，
  **test loss 0.3982，test accuracy 87.04%（10,000 样本）**；
- 模型参数量：288,746；完整 CSV、曲线、best/last checkpoint 与配置位于
  `runs/cifar10_baseline_seed42/cnn/`。

这个结果超过原计划的 80% baseline 门槛，但它只是一组 seed 的教学工程结果，没有报告
均值/标准差，也没有根据 test 反复调参。训练时 torchvision 在 NumPy 2.4 下给出一个
`VisibleDeprecationWarning`；流程与指标正常，警告不应被误写成“无任何警告”。

![CIFAR-10 BasicCNN 训练曲线](assets/cifar10_basiccnn_seed42_curves.png)

可追溯副本：[`history.csv`](assets/cifar10_basiccnn_seed42_history.csv) ·
[`config.json`](assets/cifar10_basiccnn_seed42_config.json)。大体积 checkpoint 仍只保留在被
`.gitignore` 排除的 `runs/` 中。

## 8. 项目结构

```text
project2_cifar10/
├── data.py             # CIFAR-10 划分、三档增强、synthetic 离线数据
├── models.py           # BasicCNN
├── engine.py           # train_one_epoch / evaluate
├── utils.py            # seed/RNG、JSON/CSV、可恢复 checkpoint
├── train.py            # train/validation 训练入口与 --resume
├── evaluate.py         # 独立 test 评估入口
├── plot_history.py     # CSV 曲线绘制
├── requirements.txt
├── assets/              # 正式 baseline 的曲线、CSV 与 config 小型副本
└── tests/
    └── test_smoke.py   # 离线单元与端到端 smoke tests
```

## 9. 已知边界

- 当前只跑了一个 seed，不能据此估计方差；
- `--deterministic` 仍不保证跨硬件、CUDA 和库版本逐位一致；
- 这是 W3 基础 CNN，不包含 ResNet；残差网络与系统消融留给 W4；
- 87.04% 是基础 CNN 教学 baseline，不代表 SOTA；后续提升应只看 validation 做选择，
  避免根据 test 反复调参。
