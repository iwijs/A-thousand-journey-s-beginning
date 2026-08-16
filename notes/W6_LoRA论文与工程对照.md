# W6 论文笔记：LoRA 的低秩假设怎样落到字符 GPT

论文：[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)，
Hu 等，2021。

## 1. 动机

为每个任务保存并训练一整份大模型参数，训练显存、optimizer state 和多任务存储成本都很高。
LoRA 冻结预训练权重，只注入可训练低秩分解矩阵，以更少参数表达任务更新。

本工程把这个机制缩小到 W5 的 818,048 参数字符 GPT，用来验证矩阵/冻结/merge 和 rank 对照，
而不是复现论文的大模型规模或 benchmark。

## 2. 方法

$$
W=W_0+\Delta W,
$$

$$
\Delta W=\frac{\alpha}{r}BA,
$$

其中 $W_0$ 冻结，$A,B$ 可训练。参数由 $d_{out}d_{in}$ 降为
$r(d_{in}+d_{out})$。本工程默认注入每个 block 的 QKV 和 attention output projection。

## 3. 初始化和 merge

A 使用随机初始化、B 为零，因此初始 $BA=0$，注入不改变 base logits。训练后推理可合并：

$$
W_{merged}=W_0+\frac{\alpha}{r}BA.
$$

merge 后不增加 Linear 推理路径，但不再保留独立 adapter 结构；训练/恢复应保留未 merge
checkpoint。

## 4. 本工程的实验问题

共享同一个 W5 best checkpoint 和 ROMEO/JULIET domain，比较：

| 实验 | rank | alpha | alpha/r | 可训练参数 |
|---|---:|---:|---:|---:|
| r2 | 2 | 4 | 2 | 6,144 |
| r8 | 8 | 16 | 2 | 24,576 |

主要指标是 held-out domain validation loss；step0 是 frozen base 共同参照。生成样例只作为定性
补充。

## 5. 与论文范围的差异

- 基座是小型字符 GPT，不是论文中的大规模预训练模型；
- domain 是原预训练语料的角色子集；
- 只比较两个 rank、一个 target 集合和一个 seed；
- 没有与 full fine-tuning、其他 PEFT 方法或标准下游 benchmark 系统对比；
- adapter dropout/优化器是本教学配置，不代表论文所有设置。

因此结论必须写“本次教学规模观察”，不能外推到真实 LLM 的质量/显存比例。

## 6. 工程证据

- zero-init test：注入前后 logits 完全相同；
- requires-grad 检查：只有 `lora_a/lora_b` 可训练；
- merge test：非零 adapter 合并前后 eval 输出近似相同；
- base/domain SHA-256：拒绝 adapter 与错误基座混用；
- full checkpoint 支持恢复，adapter-only checkpoint 展示 PEFT 分发方式。

## 7. 2026-08-16 实际结果

两组 step 0 validation 都是 1.4578。r2（6,144 trainable）最佳 1.4294/step 900；r8
（24,576 trainable）最佳 1.4299/step 300。到 step 1000，r8 train loss 1.3362 低于 r2 的
1.3866，但 validation 1.4552 反而高于 r2 的 1.4502。一次 seed 且最佳差 0.0005，只能说明
“更高 rank 在本次小数据实验中更快拟合 train，却没有可辨认的 validation 收益”。

正式 CUDA 首跑还发现动态创建的 adapter 默认留在 CPU；修复为显式继承 base weight 的
device/dtype，并新增 `meta`/float64 契约测试。这说明 CPU smoke 能验证数学路径，却不能覆盖
所有真实设备集成问题。

## 8. 我能否回答

1. 为什么 $BA$ 的秩最多为 $r$？
2. A/B 都为零为什么会卡住初始梯度？
3. 冻结参数为什么不等于 detach 整个 forward？
4. adapter-only 文件必须怎样绑定 base？
5. r8 更好/更差分别有哪些可能解释，什么不能据此断言？
