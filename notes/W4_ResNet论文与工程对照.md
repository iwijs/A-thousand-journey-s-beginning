# W4 论文笔记：ResNet 的主张怎样落到 `project3_resnet`

论文：[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)，
He、Zhang、Ren、Sun，2015。

## 1. 我真正要复现的主张

论文关心的不是“层数越多参数越多，所以准确率更高”，而是深层 plain network 即使没有
典型过拟合，也可能出现训练误差随深度增加反而变差的退化问题。残差学习把目标映射写成：

$$
H(x)=F(x)+x.
$$

它为信息和梯度提供显式 identity path，使额外层在不需要改变表示时更容易逼近 identity。

本工程用同一套两层 block 分别构造：

- `resnet18`：执行 $F(x)+S(x)$；
- `plain18`：只保留 $F(x)$，不加 shortcut。

因此可检验问题是：相同数据/优化配置下，residual 是否让训练更快或 validation 更好。

## 2. 论文网络与本工程的差异

论文主 ImageNet 网络接收大图，开头使用 7x7 stride-2 conv 和 max-pool。本工程面对
32x32 CIFAR-10，使用 3x3 stride-1 stem、取消初始 max-pool，避免进入 residual stages 前
过早压缩空间尺寸。

论文还分析了更深的 CIFAR 网络；本工程选择 ResNet-18 规模作为本科入门复现，不声称逐项
复现论文 ImageNet/CIFAR 表格。

## 3. projection shortcut

同 shape 时 $S(x)=x$。stage 首 block 需要通道翻倍、空间减半，使用：

$$
S(x)=W_s*x,
$$

其中 $W_s$ 是 1x1 stride-2 convolution。主分支和 shortcut 必须产生同 shape 才能逐元素加。

## 4. 工程证据

- `models.py/BasicBlock`：相加发生在第二个 BN 后、最终 ReLU 前；
- `test_residual_block_preserves_gradient_path`：主分支为零时正输入梯度沿 identity 为 1；
- `plain18`：只移除 residual，作为核心消融；
- `history.csv`：比较优化轨迹，不只比较最后一个数字；
- validation 选择 best，官方 test 不参与调参。

## 5. 2026-08-16 实际结果

同一数据与优化协议跑满 100 轮后：ResNet-18 最佳 validation 95.12%、official test 94.64%；
Plain-18 最佳 validation 94.88%、official test 94.19%。ResNet 分别高 0.24/0.45 个百分点，
差距很小。Plain 在高 lr 阶段出现更大 validation 波动，epoch 9 到 10 从 79.30% 降至
58.60%，之后随 cosine 降低逐步恢复到 94.88%。

因此本次观察支持的是“residual 的优化轨迹更稳定，并有小幅最终指标优势”，而不是“残差让
准确率大幅提升”。两个模型后期 train 都接近 100%，validation/test 差异还包含泛化因素，
不能全部归因于梯度。单 seed、略不同参数量和未锁定共有层初值，也不能代替均值/标准差与
更深网络实验。

## 6. 我能否回答

1. 退化为什么不等于过拟合？
2. $F(x)=0$ 与普通多层网络直接学 identity 有什么区别？
3. shortcut 怎样产生梯度公式中的 $I$ 项？
4. CIFAR stem 为什么不同于 ImageNet stem？
5. 当前 plain/residual 对照有哪些不完全相同的参数（projection）？
