# W4 工程实践 03：ResNet-CIFAR 复现逐行精读

> 对应源码：`2_成果作品/A-thousand-journey-s-beginning/project3_resnet`
> 配套概念速查：`W4_理论讲义_优化正则化与ResNet.md`
> 阅读目标：能从数学、shape、梯度和实验协议四个角度解释 residual block，并能独立改出
> 一个受控消融，而不是只会调用 `torchvision.models.resnet18`。

## 0. 怎么使用这份讲义

第一遍先跑测试、看调用链，不在每个公式停太久；第二遍把源码放在左侧，按本讲义的行号
逐段读；第三遍闭卷画出 BasicBlock 和一次 epoch 的状态流；第四遍亲手做 plain18 消融。

在仓库根目录执行：

```powershell
conda run --no-capture-output -n ai-learn python -m unittest discover `
  -s project3_resnet/tests -v
```

生成的 `history.csv`、checkpoint 和曲线是实验产物，不需要逐行阅读二进制 checkpoint；你要
理解的是创建它们的源码、字段语义和恢复条件。

## 1. 从 W3 到 W4：复用什么，新增什么

W3 已经回答了通用训练工程问题：

- CIFAR 官方 train 怎样固定切成 train/validation；
- augmentation 为什么只能用于 train；
- tensor 怎样统一移动到 CPU/CUDA；
- CSV、best/last checkpoint 和 RNG 怎样保存；
- test 为什么只能由独立入口最终评估。

W4 不复制这些经过测试的实现，而在 `data.py/engine.py/utils.py/plot_history.py` 中显式复用。
这样把新 bug 面集中到两个问题：模型是否真的是 CIFAR ResNet-18，以及 residual/plain 对照
是否只改变了目标变量。

```text
project2_cifar10（已验证基础设施）
       ├── data protocol ──→ project3_resnet.data
       ├── train/eval loop → project3_resnet.engine
       ├── checkpoint/RNG ─→ project3_resnet.utils
       └── CSV plot ───────→ project3_resnet.plot_history

project3_resnet 新增
       ├── models.py：BasicBlock / CifarResNet18 / plain18
       ├── train.py：SGD + momentum + ResNet 配置/消融
       ├── evaluate.py：按 checkpoint 重建模型
       └── tests：恒等梯度路径与 projection shape
```

“复用”不等于隐式魔法。`data.py` 明确列出导入名字，读者仍能沿导入跳到 W3 的唯一实现。

## 2. 残差学习的正式问题

设一个 block 理想上要学习映射 $H(x)$。普通网络直接让两层卷积逼近 $H$；ResNet 改写为：

$$
F(x)=H(x)-x,
$$

$$
H(x)=F(x)+x.
$$

它没有宣称任何函数都天然接近 identity，而是提供一条结构上的恒等路径。若额外层暂时
不需要改变表示，让残差分支逼近 $F(x)=0$ 比让多层非线性共同精确实现 $H(x)=x$ 更直接。

若忽略最后的 ReLU，梯度为：

$$
\frac{\partial L}{\partial x}
=\frac{\partial L}{\partial y}
\left(I+\frac{\partial F}{\partial x}\right).
$$

$I$ 是不经过残差分支权重乘积的直接项。它不能保证永不梯度消失/爆炸，却给优化器一条
更短的传播路径。`test_residual_block_preserves_gradient_path` 正是在一个可控极端中验证 $I$。

## 3. 五个必须先记住的 shape 契约

| 位置 | shape | 变化原因 |
|---|---|---|
| input | `(B,3,32,32)` | CIFAR RGB 图像 |
| stage1 | `(B,C,32,32)` | stride 1 |
| stage2 | `(B,2C,16,16)` | 首 block stride 2 |
| stage3 | `(B,4C,8,8)` | 首 block stride 2 |
| stage4 | `(B,8C,4,4)` | 首 block stride 2 |
| GAP | `(B,8C,1,1)` | 对空间位置取均值 |
| logits | `(B,10)` | flatten + classifier |

主分支和 shortcut 做逐元素加法，必须 shape 完全相同。若输入输出通道不同或 stride 不为 1，
identity 无法直接相加，必须由 `1x1` projection 同时改通道和空间尺寸。

二维卷积输出尺寸公式：

$$
H_{out}=\left\lfloor\frac{H_{in}+2p-d(k-1)-1}{s}+1\right\rfloor.
$$

本项目 `k=3,p=1,d=1`。`s=1` 保持尺寸，`s=2` 把偶数尺寸减半。

## 4. `__init__.py`：公开模型入口

| 行号 | 代码 | 解释 |
|---|---|---|
| L1 | 模块 docstring | 说明包的主题是 CIFAR ResNet 和 residual ablation。 |
| L3 | 导入三个模型名字 | 用户可从包顶层导入，而无需知道定义文件路径。 |
| L5 | `__all__` | 声明星号导入时的公开接口；不影响显式导入。 |

训练脚本仍从 `.models` 导入真实定义，因为内部依赖指向定义位置更容易追踪。

## 5. 四个复用文件逐行理解

### 5.1 `data.py`

L3–12 从 W3 数据模块导入增强枚举、归一化常量、synthetic Dataset、transform、三种 loader
和固定划分函数。这里没有 `*` 导入，是为了让依赖面一眼可见。

L14–24 的 `__all__` 与导入列表一致。W4 正式训练调用 `make_train_val_loaders`，训练期间
因此仍不会构造官方 test；`evaluate.py` 才调用 `make_test_loader`。

### 5.2 `engine.py`

| 行号 | 名字 | 契约 |
|---|---|---|
| L3 | `train_one_epoch` | train mode、五步更新、按样本累计 loss/accuracy。 |
| L3 | `evaluate` | eval mode + inference mode，无 optimizer 更新。 |
| L3 | `resolve_device` | `auto` 在 CUDA 可用时返回 CUDA，否则 CPU；显式错误不静默降级。 |

ResNet 比 BasicCNN 深，但训练五步没有变化：

```text
zero_grad → forward → cross entropy → backward → SGD step
```

模型复杂度属于 `models.py`，不应复制进 engine。

### 5.3 `utils.py`

L3–12 复用 W3 的 config/CSV/checkpoint/RNG。特别注意：resume 不只是
`model.load_state_dict`，还包括 optimizer momentum buffer、scheduler 位置、Python/NumPy/
PyTorch RNG 和 DataLoader generator。缺 optimizer state 时，“恢复”后的 SGD 已失去此前积累
的动量。

### 5.4 `plot_history.py`

W4 CSV schema 与 W3 相同，因此 L3 直接复用绘图函数；L7–8 保留模块命令入口。
这是用稳定接口减少重复，而不是把模型代码隐藏在另一个项目。

## 6. `models.py` L9：归一化工厂

```python
def _normalization(channels, use_batch_norm):
    return nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity()
```

| 片段 | 解释 |
|---|---|
| 前导下划线 | 表示包内部辅助函数，不是稳定公开 API。 |
| `channels` | BatchNorm2d 为每个通道维护可学习缩放/平移和 running mean/var。 |
| `nn.Identity()` | forward 原样返回输入，让后续结构无需到处写 if。 |

关闭 BN 时卷积 `bias=True`；启用 BN 时 `bias=False`，因为 BN 的平移参数已能表达偏置，卷积
bias 通常冗余。

## 7. `BasicBlock` L13–65 逐行精读

### 7.1 L13–17：类和 expansion

`BasicBlock(nn.Module)` 让子层被自动注册。`expansion=1` 表示 block 最终通道数就是传入的
`out_channels`；Bottleneck 常用 expansion=4，本项目没有 Bottleneck。

### 7.2 L18–27：构造参数

| 参数 | 含义 | 典型值 |
|---|---|---|
| `in_channels` | 输入通道 | 64/128/256 |
| `out_channels` | 输出通道 | 64/128/256/512 |
| `stride` | 第一层卷积步幅 | stage 首 block 为 2，其余为 1 |
| `use_residual` | 是否执行 shortcut 加法 | resnet18 True，plain18 False |
| `use_batch_norm` | BN 或 Identity | baseline True |

L26 `super().__init__()` 必须先初始化 `nn.Module` 的内部注册结构。L27 将 residual 开关保存为
实例状态，forward 才能据此决定是否相加。

### 7.3 L28–46：残差主分支

第一层 `3x3` 卷积使用传入 stride，因此它负责 stage 下采样；padding 1 与 kernel 3 保持
stride=1 时空间尺寸。随后 norm + ReLU。

第二层也是 `3x3`，固定 stride 1，之后只 norm、不立即 ReLU。原因是必须先把未截断的
$F(x)$ 与 shortcut 相加，再对和做 ReLU。若在相加前多一次 ReLU，就改变了 BasicBlock
定义并限制残差只能非负。

`self.relu` 被两处复用。ReLU 没有可学习参数，因此复用同一个 module 不会共享不应共享的
状态；`inplace=True` 节省部分内存，但调试 autograd 原地修改问题时要知道它的存在。

### 7.4 L48–58：shortcut

默认 `nn.Identity()`。只有同时满足 `use_residual` 且 shape 会变化时，才创建：

```text
1x1 Conv(in → out, stride) → BN/Identity
```

`1x1` 在每个空间位置做通道线性组合；stride 2 每隔一个位置取样。它不负责复杂空间特征，
主要作用是把 shape 对齐。

plain18 根本不做加法，因此没有必要构造不会使用的 projection；这使 plain 参数量略少。做
实验时要如实报告，不能写成“参数完全一样”。

### 7.5 L60–65：forward

| 行号 | 代码效果 | shape |
|---|---|---|
| L61 | conv1 → norm1 → ReLU | `(B,Cout,H/s,W/s)` |
| L62 | conv2 → norm2 | shape 不变 |
| L63–64 | residual 模式加 `shortcut(inputs)` | 两项必须完全同 shape |
| L65 | 对和做 ReLU | block 输出 |

不能写 `outputs += shortcut` 后再随意复用旧 outputs 做其他分支；原地操作可能破坏 autograd
保存的中间值。本实现使用非原地加法。

## 8. `CifarResNet18` L68–139 逐行精读

### 8.1 L71–85：配置和可变通道状态

构造函数接收类别数、基础通道、residual/BN 开关。L80 防止 `base_channels<=0` 产生难懂的
卷积错误。L83 `self.in_channels=base_channels` 是 `_make_stage` 构建期间更新的游标，不是
输入图像的固定属性。

### 8.2 L84–96：CIFAR stem 与四个 stage

stem 使用 `3x3, stride=1, padding=1`，不像 ImageNet ResNet 使用 `7x7, stride=2 + maxpool`。
对 32 像素图像沿用 ImageNet stem 会过快压缩：32 → 16 → 8，进入第一个 residual stage
前就损失大量位置细节。

四次 `_make_stage` 的通道为 $C,2C,4C,8C$，block 数都是 2；stage2–4 的首 block stride 2。
总计：stem 1 个卷积 + 8 blocks × 2 个卷积 = 17 个主分支卷积，再加分类 Linear，名称
ResNet-18 由此而来。projection 卷积不计入“18 层”传统命名。

### 8.3 L97–100：GAP 与分类器

`AdaptiveAvgPool2d((1,1))` 不依赖输入空间具体是 4x4 还是别的尺寸，逐通道取空间平均。
它把每个通道压成一个数，避免用巨大 flatten+Linear 引入大量位置相关参数。

### 8.4 L103–123：`_make_stage`

L104 先建立列表，首 block 接收 stage stride，可能负责下采样/升通道。L114 立刻把
`self.in_channels` 更新为 `out_channels`，因此后续 block 输入输出通道相同。L115 的
`range(1, blocks)` 对 blocks=2 只迭代一次，再加入一个 stride=1 block。L123 用
`nn.Sequential(*layers)` 注册并按序执行。

若把 `self.in_channels` 的更新放错位置，第二个 block 可能仍按旧通道构造，直到 forward
才报 channel mismatch。

### 8.5 L125–131：初始化

遍历 `self.modules()` 包含网络内所有嵌套子层。卷积使用 Kaiming normal，适配 ReLU；BN
的 gamma 初始化为 1、beta 为 0，使它初始近似标准化后的恒等缩放。Linear 保留 PyTorch
默认初始化。

### 8.6 L133–139：完整 forward

forward 按 stem → stage1–4 → pool → flatten → classifier。`torch.flatten(...,1)` 保留第 0
维 batch，把其余维合并；不要用无参数 `flatten()`，batch size 1 时也必须保留 batch 维。

## 9. `build_model` L142–160：让实验变量进入配置

`resnet18` 映射 `use_residual=True`，`plain18` 映射 False；其他字符串立即报错。工厂把
`base_channels` 和 BN 开关一起传入模型，保证 train/evaluate 都能由 checkpoint config
重建同一结构。

如果你新增 `wide_resnet` 却没有把 width 写进 checkpoint，保存能成功，独立 evaluate 会因
state_dict shape 不匹配失败。

## 10. `train.py` L28–89：参数、验证和目录安全

### 10.1 L28–52 `parse_args`

参数分四类：

| 类别 | 参数 |
|---|---|
| 实验身份 | model、dataset、output-dir、seed |
| 数据 | data-dir、validation-size、augmentation、batch、workers、limits |
| 模型 | base-channels、no-batch-norm |
| 优化/状态 | epochs、lr、momentum、weight decay、scheduler、device、resume |

`action="store_true"` 让 `--no-batch-norm` 出现时为 True。config 另外写入正向语义
`use_batch_norm`，避免评估时反复双重否定。

### 10.2 L55–63 `_validate_args`

CLI 能解析字符串不代表值合理。这里提前拒绝非正 epochs/batch/channels、非法 momentum、
负 weight decay，以及同时 resume/overwrite。越早失败，越不容易先下载数据或覆盖产物后
才发现错误。

### 10.3 L66–74 `_config`

`vars(args).copy()` 把 Namespace 转字典；Path 不能直接 JSON 序列化，所以逐项转字符串。
再记录 resolved device 和 selection metric。requested `auto` 与实际 `cuda` 都保留，便于
复盘环境决策。

### 10.4 L77–89 `_prepare_run`

正常新实验若发现同名产物就拒绝，防止把两次 CSV 连接成伪长实验；overwrite 只删除明确
列出的生成物；resume 要求目录已经存在且不删除任何内容。不要把整个 `runs` 递归删除当成
“方便”。

## 11. `train.py` L92–199：主流程逐段精读

### 11.1 L93–100：确定性、device、run_dir

先校验参数，再设 seed，然后解析 device。resume 时 run_dir 直接取 checkpoint 的父目录，
避免用户把恢复结果意外写到另一个实验目录。

### 11.2 L101–114：数据

传入 `pin_memory=device.type=="cuda"`。锁页内存主要帮助 CPU→GPU 异步拷贝，纯 CPU 时没有
必要开启。打印 train/validation 数量并明确 `test=not_loaded` 是防泄漏证据。

### 11.3 L115–130：模型、loss、optimizer、scheduler

模型先 `.to(device)`，engine 再把每批数据移到同一 device。CrossEntropyLoss 直接接 logits，
不在模型末尾 softmax。SGD 更新近似为：

$$
v_t=\mu v_{t-1}+g_t,
$$

$$
\theta_{t+1}=\theta_t-\eta_t v_t.
$$

cosine 学习率在 100 epochs 内从 0.1 平滑降到接近 0。scheduler 在每轮训练/验证之后 step，
CSV 记录的是该轮实际使用的学习率。

### 11.4 L135–155：resume

恢复前逐项检查 model、dataset、augmentation、seed、base_channels、BN、scheduler，以及
batch/lr/momentum/weight decay/validation/workers/limits/deterministic。随后加载 model/optimizer/
scheduler，检查 `history.csv` 最后 epoch 等于 checkpoint epoch，再恢复 best 指标、loader
generator 和全局 RNG。这样 config 不会声称使用新学习率，而 optimizer state 实际沿用旧值。

顺序很重要：模型构造会消耗随机数，所以必须在构造并载入权重之后恢复 checkpoint RNG，
否则下一轮 dropout/augmentation/shuffle 不会接上原序列。

### 11.5 L157–192：epoch 状态机

每轮依次：记录开始时间/lr → train one epoch → validation → 组 row → append CSV → 判断是否
improved → scheduler step → 保存 last → improved 时保存 best → 打印。

`best_val_accuracy` 使用严格大于，因此持平时保留较早 checkpoint。last 无条件覆盖，始终
代表最后完整完成的 epoch。两者用途不能互换：继续训练优先 last，最终推理优先 best。

### 11.6 L194–199：收尾

从 CSV 画曲线而非使用内存临时列表，证明日志本身足以重建图。参数量统计包含模型所有
参数；最终打印最佳 validation，而不是把尚未执行的 test 写成结果。

## 12. `evaluate.py` L16–53：独立 test 边界

L17–24 解析 checkpoint、数据、batch、device 等评估参数。L27 加载 checkpoint 后从 config
读取 dataset/model/base_channels/BN，构建官方 test loader 和空模型，再 load state_dict。

评估只需要 model state，不需要 optimizer；但训练 checkpoint 保留 optimizer 是为了 resume。
输出包括 selected epoch、selection metric、test loss/accuracy/sample count，让读者确认评估的
确是 validation 选出的模型和完整 10,000 样本。

## 13. `tests/test_smoke.py` 为什么有证据力

### 13.1 L18–23：两个模型的统一契约

用 base_channels=8 减少 CPU 成本，同时遍历 residual/plain，要求输出 `(2,10)`。它能抓住
stage 通道、pool/flatten、classifier 维度错误，但不能证明真实准确率。

### 13.2 L25–32：恒等梯度路径

测试把两层主分支卷积权重置零、输入设为正数并求输出和。此时：

$$
y=\operatorname{ReLU}(x)=x,
$$

$$
\frac{\partial \sum y}{\partial x}=\mathbf 1.
$$

若忘了 shortcut、错误 detach、加法位置错或 projection 不必要介入，这个断言会失败。输入用
正数是为了避开 ReLU 在负区间的零梯度。

### 13.3 L34–36：projection

`BasicBlock(8,16,stride=2)` 必须输出 `(2,16,8,8)`。这同时检查主分支和 shortcut shape 能
对齐并成功相加。

### 13.4 L38–78：端到端 CLI

临时目录保证不污染真实 runs；synthetic 和 base_channels=4 保证离线快速；子进程从
`python -m project3_resnet.train` 走真实 argparse/模块入口。测试检查五个产物、CSV 一行，
再启动独立 evaluate。它证明“能保存且能加载”，但不证明 CIFAR 指标。

## 14. 怎样设计 residual 消融

控制变量表：

| 必须相同 | 唯一变化 |
|---|---|
| 数据划分 seed | `resnet18` → `plain18` |
| augmentation | residual addition |
| epoch/batch/lr/momentum/weight decay | projection 只在 residual 模型需要 |
| BN、base channels | |
| checkpoint 选择规则 | |

至少比较：前 10 轮收敛、最佳 validation、达到最佳的 epoch、train/val gap、总时间和一次 test。
如果 plain 参数略少，应在表中报告，不把结构差异藏起来。

不要先看 test 再决定训练多少轮。训练策略由 validation 决定，test 只在方案冻结后评一次。

## 15. 曲线诊断

- train/val 同时高 loss：欠拟合或仍在优化早期，先看学习率和训练时间；
- train 持续降、val 先降后升：过拟合，best checkpoint 应早于 last；
- val 大幅来回跳：检查 batch statistics、学习率过高、validation 是否误用随机增强；
- residual 明显更快而最终接近：证据支持“优化更容易”，不等于表示能力一定更强；
- 两者差距很小：单 seed/网络深度/训练配置可能不足以显现，不应捏造论文式结论。

## 16. 常见错误

### shortcut shape mismatch

打印主分支和 shortcut shape。只要 stride 或通道变化，就不能用纯 identity。

### 关掉 residual 却仍加 projection

plain forward 应完全不执行 shortcut。否则它仍有一条线性旁路，不是普通深层网络。

### 忘记 `model.eval()`

BN 会继续用当前 batch 统计并更新 running state，validation 会依赖 batch 组成。

### resume 后学习率跳变

确认加载了 scheduler state，且 `--epochs` 仍是原计划总 horizon；任意延长 cosine horizon 并
不等价于原实验。

### 把最好 test 当选模指标

检查 config/checkpoint 的 `selection_metric` 必须是 `validation_accuracy`。

## 17. 闭卷自测

1. 为什么 CIFAR stem 不照搬 ImageNet 的 7x7+maxpool？
2. BasicBlock 第二个 norm 后为什么先相加、再 ReLU？
3. 写出 stage2 首 block 主分支与 shortcut 的完整 shape。
4. identity shortcut 怎样在梯度公式中产生 $I$ 项？
5. `self.in_channels` 在 `_make_stage` 中为什么必须更新？
6. 为什么 GAP 后 classifier 输入是 `8*base_channels`？
7. BN 开启时卷积 bias 为什么可省？
8. residual/plain 消融哪些配置必须相同？
9. best 与 last 各用于什么？
10. resume 为什么必须恢复 momentum、scheduler、RNG 和 loader generator？
11. 恒等梯度测试为什么使用正输入和零主分支权重？
12. 从 CLI 开始口述到第一个 SGD step 的跨文件调用链。

## 18. 实践任务

- [ ] 画出一个 `64→128,stride=2` block 的双分支 shape。
- [ ] 手算该 projection 的 `1x1` 权重参数量。
- [ ] 跑 4 项 smoke tests 并解释每项不能证明什么。
- [ ] 完成 resnet18 baseline，只根据 validation 选择 best。
- [ ] 完成 plain18 单变量消融。
- [ ] 用同一 prompt 式表格记录配置、参数、最佳 val、一次 test、时间。
- [ ] 能在不看源码时写出 BasicBlock forward 的四行伪代码。

达到这些要求，才是“理解并复现 ResNet 的核心主张”，而不只是把一个深网络训练脚本跑完。

## 19. 2026-08-16 正式实验记录

固定环境/config 见项目 README。ResNet-18 100 轮实测：

| 指标 | 结果 |
|---|---:|
| 参数量 | 11,173,962 |
| 最佳 validation accuracy | 95.12% |
| 最佳 epoch | 99 |
| epoch 100 validation | 95.04% |
| official test loss（epoch 99 best） | 0.1989 |
| official test accuracy | 94.64% |

训练早期 lr 接近 0.1 时 validation 曾在相邻轮间明显波动；cosine 降到约 0.04 后首次越过
90%，低于 0.01 后逐步到 95%。后期 train accuracy 接近 100%，validation 约 95%，形成明确
generalization gap。最终 test 只评 validation 选出的 epoch 99 一次。

plain18 使用完全相同数据划分、增强、SGD、scheduler、epochs、seed 和 deterministic 设置。
还要注意：同一个 seed 只让每个程序的随机过程可复现。ResNet 多构造了 projection 参数，会多
消费一部分 RNG 状态，因而两种架构后续共有层的初始张量并不保证逐元素一致。当前实验控制了
训练协议，但不是“锁死所有主分支初值”的配对实验；单 seed 差异只能记为本次运行观察。

### 19.1 完整对照结果

| 指标 | ResNet-18 | Plain-18 | ResNet 减 Plain |
|---|---:|---:|---:|
| 参数量 | 11,173,962 | 11,000,138 | +173,824 |
| 最佳 validation accuracy | 95.12%（epoch 99） | 94.88%（epoch 100） | +0.24 pp |
| official test accuracy | 94.64% | 94.19% | +0.45 pp |
| official test loss | 0.1989 | 0.2402 | -0.0413 |
| 100 轮累计训练时间 | 75.4 min | 61.2 min | +14.2 min |

这组结果不支持“18 层时 residual 会带来巨大最终准确率提升”这种夸张表述。带 BN、充分训练的
Plain-18 最终也达到很高性能；本次单 seed 的 test 差距只有 0.45 个百分点。更明显的是优化
轨迹：Plain-18 在 lr 仍接近 0.1 时 validation 剧烈波动，epoch 9 到 10 从 79.30% 跌到
58.60%，直到 cosine 把 lr 降低后才逐渐追上。ResNet 也有 validation 波动，但没有这次 Plain
运行中的同等幅度崩落。

因此正确结论是：本次教学规模实验观察到 residual 带来小幅最终指标优势和更稳定的高学习率
优化，但计算/参数略增；要判断平均效果，仍需多个 seeds、更深 plain network，以及锁定共有
主分支初值的配对实验。曲线中的真实反例比一句“ResNet 一定更好”更值得学习。
