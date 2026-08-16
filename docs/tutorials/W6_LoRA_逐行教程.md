# W6 工程实践 05：LoRA 参数高效微调逐行精读

> 对应源码：`2_成果作品/A-thousand-journey-s-beginning/project5_lora`
> 前置工程：`project4_nanogpt` 的 validation-selected `best.pt`
> 配套概念速查：`W6_理论讲义_实验复现与LoRA.md`
> 目标：能推导 LoRA 参数量与前向公式、解释零初始化/冻结/merge，并独立完成一个控制变量
> rank 对照，不把“小模型 adapter 实验”夸大成工业大模型微调。

## 0. W6 为什么接在 W5 后面

W5 从随机初始化训练整个 GPT，所有参数都可更新；W6 从 W5 checkpoint 出发，把基座冻结，
只学习很小的低秩更新。这让“预训练 → 领域适配”成为一条可执行链，而不是两个无关 demo。

```text
Tiny Shakespeare
      ↓ W5 full-parameter pretraining
base best.pt
      ↓ freeze base + inject LoRA
ROMEO/JULIET speeches
      ↓ W6 adapter training
LoRA best.pt / adapter_best.pt
```

这里的领域语料本来就是预训练语料子集，所以实验只能说明“重新加权角色风格的优化行为”，
不能声称获得新知识或跨域泛化。

## 1. 从 full fine-tuning 到 low-rank update

设原线性层权重：

$$
W_0\in\mathbb R^{d_{out}\times d_{in}}.
$$

全量微调直接学习任意同 shape 更新 $\Delta W$，参数数为 $d_{out}d_{in}$。LoRA 假设有用更新
落在低维子空间，用：

$$
A\in\mathbb R^{r\times d_{in}},\qquad
B\in\mathbb R^{d_{out}\times r},
$$

$$
\Delta W=\frac{\alpha}{r}BA.
$$

前向：

$$
y=xW_0^\top+b+\frac{\alpha}{r}xA^\top B^\top.
$$

训练参数数：

$$
N_{LoRA}=r(d_{in}+d_{out}).
$$

压缩有意义的条件是：

$$
r(d_{in}+d_{out})\ll d_{in}d_{out}.
$$

rank 不是“模型有几个 head”，而是允许权重更新矩阵最多有多高的秩。

## 2. 为什么 A 随机、B 为零

项目初始化：

$$
A\sim\text{Kaiming},\qquad B=0.
$$

所以注入瞬间 $BA=0$，输出严格等于 base。反向初始时：

$$
\frac{\partial L}{\partial B}\propto
\frac{\partial L}{\partial y}(xA^\top)^\top,
$$

一般非零；而 $\partial L/\partial A$ 初始含 $B$，可能为零。第一次更新先让 B 离开零，后续
A 也获得梯度。如果 A/B 都为零，两边梯度都会卡住；如果两者都随机，注入瞬间会扰动 base。

测试用 `torch.equal` 检查注入前后 logits 完全一致，而不只是“大致接近”。

## 3. `LoRAConfig` L13–28

| 字段 | 作用 | 默认 |
|---|---|---:|
| rank | 低秩瓶颈 $r$ | 8 |
| alpha | update 缩放分子 | 16 |
| dropout | 只作用于 adapter 输入 | 0.05 |
| target | attention 或 attention+MLP | attention |

frozen dataclass 防止运行中改配置。`__post_init__` 拒绝非正 rank/alpha、非法 dropout 和未知
target。`to_dict` 进入 checkpoint，加载时不依赖自定义类反序列化。

比较 rank 2/8 时令 alpha 4/16，使 $\alpha/r=2$ 相同；这样唯一主要变化是 rank/参数容量，
而不是 update 的显式缩放倍数。

## 4. `LoRALinear` L31–69 逐行精读

### 4.1 L34–50 构造

`base` 是原 `nn.Linear`，完整保留权重/bias。保存 rank/scaling，建立 dropout、
`lora_a: din→r` 和 `lora_b: r→dout`。`factory_kwargs` 显式继承 `base.weight` 的 device 与
dtype；否则在 CUDA base 中动态注入会得到 CPU adapter，并在第一次矩阵乘法报跨设备错误。
A Kaiming、B zeros。最后遍历 base 参数并将 `requires_grad=False`。

注意 base 仍是子模块，所以完整 state_dict 会包含 `...base.weight`；冻结只阻止 optimizer
更新，不会从保存/forward 中删除它。

### 4.2 L52–54 forward

```python
base(inputs) + lora_b(lora_a(dropout(inputs))) * scaling
```

输入可有任意前导维，例如 GPT hidden `(B,T,C)`；Linear 只作用最后一维，LoRA 路径自然得到
`(B,T,dout)`，与 base 相加。

dropout 只进入 adapter 路径，base 输出保持确定；eval 模式 dropout 关闭。

### 4.3 L56–69 `merged_linear`

创建与 base 同 in/out/bias/device/dtype 的普通 Linear。在 no_grad 中计算：

$$
W_{merged}=W_0+\frac{\alpha}{r}BA.
$$

复制 bias 后返回。merge 后推理只需一次普通 Linear，不再执行两条路径；但 adapter 的可分离
结构和 optimizer state 丢失，不应用 merge 后模型继续按同一方式训练。

若 adapter dropout 非零，merge 等价只应在 eval 模式讨论：训练模式的随机 dropout 无法折叠
进固定矩阵。

## 5. 递归注入 L72–95

### 5.1 `_replace` 怎样遍历

`list(module.named_children())` 先固定当前 children 快照，避免遍历时替换子模块导致迭代器
变化。`full_name` 用点连接父路径，最终类似：

```text
blocks.0.attention.qkv
blocks.0.attention.projection
```

attention target 通过路径后缀精确识别；all-linear 还匹配 `.feed_forward.layers.` 中的 Linear。
lm_head 不在目标中，避免破坏与 token embedding 的 tied weight。

命中时 `setattr(module,name,LoRALinear(...))` 原位替换并记录路径；否则递归进入 child。

### 5.2 `inject_lora`

先冻结模型全部参数，再递归替换。新建 A/B 默认 requires_grad True，形成唯一可训练集合。若
没有找到目标立即报错，防止模型命名变化后“训练成功但实际 0 个 adapter”。

## 6. `merge_lora` L98–106

同样递归遍历 children；遇到 LoRALinear 就用 `merged_linear()` 替换，遇到普通 child 继续
递归并给返回路径加父前缀。返回列表供调用者确认到底 merge 了哪些层。

测试随机化 B 后，对同一输入比较 merge 前后 eval logits，容差 $10^{-6}$。若忘 scaling、
矩阵顺序写成 $AB$ 或 transpose 错，测试会失败。

## 7. adapter state 与参数统计 L109–120

`adapter_state_dict` 从完整 state_dict 只筛名字含 `.lora_a.`/`.lora_b.` 的 tensor，并
detach+CPU，得到小型可移植权重。它必须和 base checkpoint SHA、模型结构、注入配置一起
保存，否则单独 tensor 无法知道应装到哪里。

`parameter_counts` 分别统计全部与 requires_grad 参数。W6 启动日志打印：

```text
trainable / total_with_adapters / fraction
```

total 含 frozen base 和新增 adapter，trainable 只含 A/B。

## 8. `prepare_domain.py` L12–40

### L12–20 `extract_speeches`

Tiny Shakespeare 用空行分隔 speaker blocks。函数按 `\n\n` 切块，取第一行；只有第一行以
冒号结尾且 speaker 在集合中才保留。最后用双换行重新连接并补末尾换行。

如果一个也没找到就报错，避免写出空训练文件。这个简单 parser 依赖当前语料格式；换 corpus
不能假定通用。

### L23–40 CLI

默认 source 是 W5 corpus，output 是 `romeo_juliet_speeches.txt`，speaker 默认 ROMEO/JULIET。
已有文件不 overwrite 时报告长度/hash。正式得到 49,563 字符和固定 SHA-256。

为什么不随便写一份中文语料？W5 字符表没有中文，CharTokenizer 会报 unknown；要跨文字系统
必须重新设计 tokenizer/embedding，而不是在 LoRA 实验里偷偷改变词表。

## 9. `train.py` L24–52：哈希与 CLI

`file_sha256` 分块读取 base checkpoint，避免一次把大文件全部读入内存。domain 用文本 UTF-8
hash，base 用文件 bytes hash；二者语义不同。

CLI 必须给 `--base-checkpoint`。其余包含 domain/output、steps/eval、batch/lr、rank/alpha/
dropout/target、device/seed/resume。正式 rank 对照除 rank/alpha/output-dir 外保持一致。

## 10. `train.py` L56–101：日志、评估和保存

### CSV

`append_history` 与 W5 schema 一致，便于复用 plot。`last_logged_step` 是 resume 一致性检查。

### estimate_loss

与 W5 相同：eval + inference mode，对 train/val 各平均随机 batches，最后恢复 train mode。
这里 validation 是 domain corpus 后 10%，不是 W5 原 validation。

### save_state

完整训练 checkpoint 保存 metadata、完整 model state、adapter-only state、optimizer、step、best
val、selection metric 和 RNG。先写 `.tmp` 再原子 replace。

为什么完整 state 里重复 base？为了教学阶段 resume/独立 generate 简单可靠；同时提供
`adapter_best.pt` 展示真实 PEFT 部署思想。严肃大模型场景通常只分发 adapter 和 base 标识。

## 11. `train.main` L104–249 逐段精读

### 11.1 L105–130：载入 base、domain 与对齐 RNG

校验参数，设 seed/device，用 `weights_only=True` 加载 W5。tokenizer 直接来自 base characters。
domain encode 遇未知字符失败；连续 90/10 split；validation 必须长于 base block_size。

然后由 base model_config 重建 CharGPT、load base state，再 inject LoRA。顺序不能反：若先注入，
state_dict 路径已从 `qkv.weight` 变成 `qkv.base.weight`，原 W5 state 无法直接加载。不同 rank 的
A 矩阵尺寸不同，初始化时会消费不同数量的随机数，所以注入后再次 `set_seed`：它不重写已经
初始化的参数，只让两组的数据抽样与 Dropout 从同一 RNG 状态出发。resume 随后恢复 checkpoint
RNG，因此中断续训仍接回原随机流。

### 11.2 L131–148：optimizer 与目录

optimizer 参数列表显式筛 `requires_grad`，即使误把 frozen 参数传入通常也不更新，但显式筛选
减少 optimizer state/歧义。run_dir 与 overwrite/resume 规则延续前几周。

### 11.3 L149–178：metadata/config

metadata 是 checkpoint 兼容身份：base hash/path、model config、characters、domain hash、LoRA
config、被替换模块列表，以及 eval/log interval、batch、lr、weight decay、seed/deterministic
组成的 training config。config 再加 resolved device、trainable/total/fraction、domain 长度和 CLI。

这使实验不是“某个 best.pt”，而是能回答：哪个基座、哪份数据、哪些层、多少 rank、多少参数。

### 11.4 L182–202：resume

比较 base hash、domain hash、LoRA config、model config 和 training config；再加载 full model/
optimizer，检查 CSV step，恢复 best/RNG。路径相同但 base 内容变化或命令行学习率变化都会拒绝。

### 11.5 L208–247：微调循环

step 0 先测 frozen base 在 domain 的 loss；这也是 rank2/rank8 公共起点。evaluation 后保存
last，改进时保存 full best 和 adapter_best。训练 batch 仍用 W5 block_size，forward/loss 后
只清零/反传/clip/step trainable adapter 参数。

base `requires_grad=False` 意味着 autograd 不为其叶子参数累积 grad；但梯度仍必须经过 base
运算传播到 LoRA 分支相关计算，冻结不等于把整个层 `detach()`。

## 12. `generate.py` L18–47

从 W6 checkpoint 读 model config/characters/LoRA config。先构造普通 CharGPT，再按完全相同
配置 inject，使 state_dict 路径匹配，最后 load full model state、eval、encode prompt、调用 W5
的 generate。

同 prompt/temperature/top-k/seed 才能公平比较 W5 base、r2、r8。只看三段随机文本很容易把
采样噪声误认成 adapter 效果，所以主要量化指标仍是 held-out domain validation loss。

## 13. 五项测试的证明边界

### 零初始化等价

随机小 GPT 注入 attention LoRA 前后 logits `torch.equal`，并检查只 LoRA 参数 trainable。证明
初始化和冻结正确，不证明训练能提高 validation。

### device/dtype 继承

用 `meta` device、float64 base 构造 LoRALinear，检查 A/B 与 base 完全同 device/dtype。这个测试
来自正式 GPU 首跑暴露的真实 bug：只在 CPU float32 上做 forward smoke 无法发现动态新层默认
落在 CPU 的问题。

### merge 等价

把 B 人工设为非零，eval 下比较 merge 前后 logits allclose。证明矩阵组合/缩放正确，不证明
量化或不同 dtype 下始终无误差。

### domain extraction

三段玩具 blocks 只保留 ROMEO/JULIET，证明筛选基本格式，不证明能解析所有戏剧文本变体。

### 完整 CLI

临时 W5 checkpoint + 同词表 corpus 分别用 rank 2/rank 4 跑 4 steps，检查
config/history/best/last/adapter/curve、验证两者 step 0 loss 完全相同，并独立生成。证明接口
闭环和 RNG 对齐协议，不代表正式领域适配质量。

## 14. rank 对照怎样保持控制变量

| 项目 | r2 | r8 |
|---|---:|---:|
| base checkpoint hash | 相同 | 相同 |
| domain hash/split | 相同 | 相同 |
| seed/batches | 42 | 42 |
| target modules | attention | attention |
| rank | 2 | 8 |
| alpha | 4 | 16 |
| alpha/r | 2 | 2 |
| dropout/lr/steps/eval | 相同 | 相同 |

注入之后重新设 seed，使 step 0 的随机 evaluation batches、后续数据抽样和相同 shape 的
Dropout mask 从同一 RNG 状态出发；但不同 rank 的矩阵运算与浮点累积仍不可能逐 bit 对齐。
固定 seed 的目的主要是减少无关差异，严谨结论仍需多个 seeds。

应报告：

- step0 base domain loss；
- 最佳 validation loss 与 step；
- trainable 参数和比例；
- wall time；
- train/val gap；
- 固定采样配置的文本作为定性补充。

## 15. 怎样解释可能结果

### r8 val loss 更低

本次配置下更高容量可能更好拟合 domain，但单 seed 不证明 rank 越高普遍越好。检查 trainable
参数代价和是否开始过拟合。

### r2 接近 r8

角色子域可能只需很低秩更新；也可能训练步数/数据不足以显现差异。可作为后续假设，不是
最终定律。

### train 降、val 升

小 corpus 过拟合。优先根据 validation 选择较早 best；可研究 dropout/更少 steps，但每次只
改一个变量。

### step0 两组不同

这是实现或评估随机性问题。相同 base/domain/seed/eval batches 理应有相同起点；先修复再比较。

## 16. 常见错误

### base 实际还在训练

打印所有 `requires_grad=True` 名字，必须只含 `lora_a/lora_b`。检查 optimizer 参数集合。

### B 零初始化后模型仍变化

确认 LoRA dropout 不影响 base 路径；B 真为零；模型比较时两次都处于同一 eval/train 状态。

### merge 后输出不等

检查 $B@A$ 顺序、alpha/r scaling、bias 和 dtype/device；在 eval 模式比较。

### CUDA base 与 CPU adapter 冲突

动态创建 A/B 时传入 base weight 的 `device`/`dtype`，不能依赖 `model.to(cuda)`：注入发生在
基座移动到 GPU 之后，新子层不会被过去已经执行完的 `.to()` 自动补迁移。

### adapter 加载到错 base

比较 base SHA-256、model config、characters、target/rank；不能只凭文件名。

### 领域语料出现未知字符

当前 tokenizer 没有 `<unk>`。要么清理到 base 字符表，要么明确扩词表并处理 embedding/head，
后者已超出“只改 LoRA”控制变量。

## 17. 闭卷自测

1. 写出 $W_0+\alpha BA/r$ 的各矩阵 shape。
2. 推导 full 与 LoRA 参数量。
3. 为什么不能 A/B 都初始化为零？
4. B=0 时为什么输出严格等于 base？
5. `requires_grad=False` 与 `detach()` 有什么区别？
6. 为什么默认不替换 tied lm_head？
7. injection 为什么必须在 load W5 state 之后？
8. merge 的矩阵顺序为什么是 $B@A$？
9. adapter-only 文件必须携带哪些元信息？
10. 为什么 step0 是关键 baseline？
11. rank2/rank8 除 rank/alpha 外哪些配置必须相同？
12. 为什么相同 alpha/r 仍不代表两个模型更新能力相同？
13. domain 是预训练子集对结论有什么限制？
14. 从 W5 best.pt 口述到第一次 LoRA optimizer step。
15. merge 后为什么适合推理、不适合继续同一 adapter 训练？

## 18. 最终验收

- [ ] 手算 attention qkv/projection 的 LoRA 参数量。
- [ ] 解释并运行 5 项 smoke tests。
- [ ] 验证启动日志中只有 adapter 可训练。
- [ ] 完成 r2/r8 两组共享 base/domain/seed 的实验。
- [ ] 表格报告参数比例、step0、best val、step、时间。
- [ ] 用固定采样配置保存 base/r2/r8 样例。
- [ ] 结论明确使用“本次教学规模实验观察”，不写成普适规律。
- [ ] 能区分 full checkpoint、adapter-only checkpoint 和 merged model。

真正的完成标准不是“用了 LoRA 这个热点词”，而是能说明低秩假设如何变成矩阵、哪些参数
获得梯度、实验对照支持什么，以及为什么当前结果不能直接外推到大模型。

## 19. 2026-08-16 正式实验记录

共享 W5 step-5000 best、同一 domain/hash、attention target、1000 updates、batch 32、lr 0.001、
dropout 0.05、seed 42。两组结果：

| 指标 | rank 2 / alpha 4 | rank 8 / alpha 16 |
|---|---:|---:|
| trainable / total | 6,144 / 824,192 | 24,576 / 842,624 |
| trainable fraction | 0.75% | 2.92% |
| step 0 train / val | 1.4374 / 1.4578 | 1.4374 / 1.4578 |
| best validation | **1.4294**（step 900） | 1.4299（step 300） |
| step 1000 train / val | 1.3866 / 1.4502 | 1.3362 / 1.4552 |
| elapsed | 33.8 s | 33.5 s |

step 0 完全相同验证了 $B=0$ 与重新对齐 RNG 的组合协议。r8 的最终 train loss 更低，却没有
得到更低 validation；它的 best 更早出现在 step 300，之后 validation 总体回升。这符合“小数据
上容量增加可能更快过拟合”的解释，但 r2/r8 最佳只差 0.0005，远小于评估波动，不能据此宣称
rank 2 普遍优于 rank 8。

固定生成样例中，r8 更频繁输出 ROMEO/JULIET 标签，r2 仍出现其他角色；两者都有伪词、语法
错误与长程不一致。生成只提供可观察的风格变化，held-out validation 才是预先约定的主指标。
此外，正式 GPU 首跑发现 adapter 默认在 CPU 的 bug，修复为继承 base device/dtype，并把该
失败模式加入第五项测试；这正是“CPU smoke 通过”仍不能替代真实 GPU smoke 的工程教训。
