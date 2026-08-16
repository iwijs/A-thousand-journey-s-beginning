# W5 工程实践 04：字符级 GPT 从零实现逐行精读

> 对应源码：`2_成果作品/A-thousand-journey-s-beginning/project4_nanogpt`
> 配套概念速查：`W5_理论讲义_注意力Transformer与GPT.md`
> 目标：读完后能闭卷写出单头 scaled dot-product attention、解释 causal mask 的信息边界，
> 并沿 shape 走完 tokenizer → batch → GPT → loss → backward → generation。

## 0. 阅读顺序

1. 先读 `data.py`，确认“标签”怎样由文本自动产生；
2. 精读 `models.py`，每行旁边写 shape；
3. 读 `train.py`，分清 optimizer step、evaluation step 和 checkpoint step；
4. 读 `generate.py`，理解训练的并行预测怎样变成推理的逐 token 循环；
5. 最后读 tests，用测试反证最常见的伪正确实现。

先跑：

```powershell
conda run --no-capture-output -n ai-learn python -m unittest discover `
  -s project4_nanogpt/tests -v
```

## 1. 语言模型的数据不是“输入句子 + 人工类别”

对字符序列 $x_1,\ldots,x_{T+1}$：

$$
X=(x_1,\ldots,x_T),\qquad Y=(x_2,\ldots,x_{T+1}).
$$

同一 context 同时产生 $T$ 个监督信号。模型输出 logits shape `(B,T,V)`，targets 是 `(B,T)`。
flatten 后交叉熵相当于把 $B\times T$ 个 next-character 分类问题一起平均：

$$
L=-\frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T}
\log p(x_{b,t+1}\mid x_{b,1:t}).
$$

causal mask 保证第 $t$ 个预测只使用前缀 $x_{1:t}$。没有 mask 时模型会偷看 target 后面的
字符，训练 loss 看起来很好，生成却无法复现训练条件。

## 2. 全局 shape 字典

| 符号 | 含义 | 默认值/shape |
|---|---|---|
| $B$ | batch size | 64 |
| $T$ | context length | 128 |
| $V$ | 字符表大小 | 由语料决定 |
| $C$ | embedding width | 128 |
| $H$ | head 数 | 4 |
| $d_h$ | 每头维度 $C/H$ | 32 |
| token ids | 整数索引 | `(B,T)` |
| hidden | 连续表示 | `(B,T,C)` |
| Q/K/V | 分头后 | `(B,H,T,d_h)` |
| attention scores | 每头位置两两相关 | `(B,H,T,T)` |
| logits | 每位置字符分数 | `(B,T,V)` |

遇到 attention bug 时先检查 shape，不要先调学习率。

## 3. `data.py` L10–32：`CharTokenizer`

### 3.1 L11–15 初始化

`characters` 必须非空且唯一。保存有序列表，并用 enumerate 建 `stoi`。字符顺序决定 token
ID；同一 checkpoint 必须保存并恢复完全相同的列表，不能每次靠另一个 corpus 重建。

### 3.2 L17–19 `from_text`

`sorted(set(text))` 先去重再排序，使同一文本得到确定性词表。字符级 token 可能包括换行、
空格、标点和大小写；它们都是不同 token。

### 3.3 L21–25 `encode`

先计算 prompt 中不在词表的字符，存在时明确报错；然后逐字符映射整数。这里不设置 `<unk>`，
是为了让教学实验在词表不一致时立即失败，而不是静默改变文本。

### 3.4 L27–32 `decode` 和 vocab size

decode 用 token id 索引 `characters` 并拼接。正确性契约：

```python
decode(encode(text)) == text
```

property 让调用者写 `tokenizer.vocab_size`，但每次仍由字符列表长度计算，不维护重复状态。

## 4. `data.py` L35–53：加载、编码和连续切分

`load_corpus` 检查文件存在和最小长度。错误提示直接给出 prepare_data 命令，降低环境问题的
定位成本。UTF-8 明确写出，避免 Windows 默认编码改变语料。

`encode_and_split` 要求 fraction 在 $(0,1)$，用完整文本建立 tokenizer，再转 `torch.long`。
Embedding 的索引必须是整数 long。前 90% train、后 10% validation 是连续切分，而不是随机
打散字符；随机打散会破坏语言顺序，也可能让相邻片段跨 split 泄漏。

用全语料建字符表只暴露“有哪些字符”，不暴露 validation 的字符顺序目标；若研究严格的
未知字符泛化则需只用 train 词表并设计 `<unk>`，不属于本教学范围。

## 5. `data.py` L56–70：`get_batch`

| 行号 | 动作 | 结果 |
|---|---|---|
| L63–64 | 检查 1D 且长度大于 block | 至少能取一个完整输入+目标 |
| L65 | 随机采样 B 个起点 | `(B,)` |
| L66 | 每个起点切 `T` 个 input | stack 为 `(B,T)` |
| L67 | 起点右移一位切 target | `(B,T)` |
| L68 | 一起移动到 device | 模型/数据同设备 |

`torch.randint` 使用全局 PyTorch RNG，所以 checkpoint 恢复 RNG 后，下一批随机 context 能接
上中断前序列。上界是 `len(data)-block_size`（exclusive），确保 target 的最后索引合法。

## 6. `GPTConfig` L14–33：结构是 checkpoint 的一部分

frozen dataclass 防止构造后意外改字段。`__post_init__` 检查正数、dropout 范围和
`n_embd % n_head == 0`。最后一项保证能把 C 等分为 H 个 head。

`to_dict()` 使用 `asdict`，checkpoint/JSON 不直接依赖 Python dataclass 对象，`weights_only`
加载也更安全。

## 7. `CausalSelfAttention` L36–60 逐行精读

### 7.1 L37–47 构造层和 mask

`qkv = Linear(C,3C)` 一次矩阵乘法同时产生 Q/K/V，再沿最后一维三等分；这与三个独立
Linear 数学上等价，但实现更紧凑。

`projection = Linear(C,C)` 在各 head 拼回后混合信息。两个 dropout 分别作用于 attention
weights 和 residual 输出。

`torch.tril(ones(T,T,bool))` 产生下三角 True：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

reshape 为 `(1,1,T,T)` 后，前两个 1 可广播到 batch/head。`register_buffer` 让 mask 随
model `.to(device)` 和 state_dict 移动/保存，但不成为可训练 Parameter。

### 7.2 L49–54：Q/K/V 分头

输入 `(B,T,C)` 经过 qkv 得 `(B,T,3C)`；split 后每个 `(B,T,C)`。先 view
`(B,T,H,d_h)`，再 transpose 1/2 得 `(B,H,T,d_h)`。

为什么 transpose 而不是仅 view？view 只重解释连续内存顺序，不能把 head 维真正移动到
矩阵乘法需要的位置。后续每个 batch/head 独立做 `T×d_h` 与 `d_h×T`。

### 7.3 L55：scaled dot product

$$
S=QK^\top/\sqrt{d_h}.
$$

shape `(B,H,T,T)`。若 q/k 各分量方差近似 1，点积方差随 $d_h$ 增大；除以
$\sqrt{d_h}$ 使尺度稳定，避免 softmax 过早饱和、梯度过小。

### 7.4 L56–57：mask 与 softmax

只切 `:sequence_length`，所以同一最大 mask 支持任何 $T\le block\_size$。对未来位置填
$-\infty$，softmax 后其概率严格为 0。不能先 softmax 再把未来权重乘 0 而不重新归一化，
否则保留位置概率和小于 1。

softmax 必须沿最后的 key-position 维 `dim=-1`，含义是“每个 query 在可见 keys 上分配
总和为 1 的权重”。

### 7.5 L58–60：聚合 V、拼回 heads

`weights @ value`：`(B,H,T,T) @ (B,H,T,d_h) -> (B,H,T,d_h)`。transpose 回
`(B,T,H,d_h)` 后内存通常不连续，必须 `.contiguous()` 再 view `(B,T,C)`。最后 projection
和 residual dropout，输出 shape 与输入一致，才能做 residual addition。

## 8. `FeedForward` L63–74

每个 token 独立经过：

$$
\mathbb R^C\to\mathbb R^{4C}\to\mathbb R^C.
$$

Linear 在 `(B,T,C)` 上自动把前两维当 batch，只变最后一维。GELU 比硬 ReLU 平滑；第二个
Linear 把宽度投回 C，以便 residual 相加。attention 做 token 之间通信，MLP 在每个 token
内部变换通道，两者职责不同。

## 9. `TransformerBlock` L77–87

构造两个独立 LayerNorm，分别供 attention 和 MLP 使用。forward：

```python
x = x + attention(norm1(x))
x = x + feed_forward(norm2(x))
```

这是 pre-LN。残差主路保持未经 norm 的 x，深层训练通常比 post-LN 更稳定。两个 `+` 都
不能省，也不能把 `norm1`/`norm2` 错共享为同一个有参数层。

LayerNorm 对每个 token 的 C 个通道归一化，不依赖 batch 其他样本；这与 CNN 的 BatchNorm
统计维度和 train/eval 行为不同。

## 10. `CharGPT.__init__` L90–101

### 两种 embedding

token embedding 查表 `(B,T)->(B,T,C)`，表示“是什么字符”；position embedding 查
`0..T-1 -> (T,C)`，表示“在第几个位置”。二者相加让模型区分相同字符出现在不同位置。

### blocks 与 final norm

`ModuleList` 注册 n_layer 个独立 block；普通 Python list 不会自动注册参数。forward 需要
显式循环。final LayerNorm 在输出 head 前稳定表示。

### weight tying

`lm_head.weight = token_embedding.weight` 让输入字符表示矩阵和输出分类矩阵共享同一 Parameter，
减少 $V\times C$ 参数并建立输入/输出表示联系。赋值发生在初始化前；`apply` 遍历时同一
共享 Parameter 可能被初始化路径触及，但最终仍是同一个存储。

## 11. 初始化 L103–110

Linear/Embedding 权重正态分布 std 0.02，Linear bias 为零。初始化不是理论唯一选择；它是
需要写进可复现实现的优化条件。模型 config 与 seed 一起决定初始参数。

## 12. `CharGPT.forward` L112–131

### L117–123：输入与表示

读取 B/T，拒绝超过 block_size。`arange(T, device=token_ids.device)` 避免 position ids 留在
CPU。token `(B,T,C)` 与 position `(T,C)` 按 batch 维广播相加，再 dropout。

### L124–126：堆叠 block 与输出

hidden 依次经过所有 blocks，shape 始终 `(B,T,C)`；final norm 后 lm_head 得 `(B,T,V)`。

### L127–131：可选 loss

targets 为 None 时只返回 logits，供 generation；有 targets 时把 logits reshape
`(B*T,V)`、targets reshape `(B*T)` 交给 CrossEntropyLoss。使用 reshape 而不是 view，
因为前面的 transpose/计算可能影响连续性。

模型末尾不加 softmax：cross entropy 内部使用数值稳定的 log-softmax。生成时才显式 softmax。

## 13. `generate` L133–159

`@torch.inference_mode()` 禁止构图，节省显存；调用者仍应 `model.eval()` 关闭 dropout。

每一步：

1. 截取最后 block_size 个已有 token；
2. forward 得所有位置 logits；
3. 只取最后位置预测下一个 token；
4. 除 temperature；
5. 可选 top-k，把其余 logits 置 $-\infty$；
6. softmax 变概率并 multinomial 采样；
7. 拼到原序列，重复。

temperature：

$$
p_i=\operatorname{softmax}(z_i/\tau).
$$

$\tau<1$ 更尖锐，$\tau>1$ 更平；必须大于 0。当前实现每步重新计算整个 context，复杂度高，
但最容易验证；KV cache 留作后续推理优化。

## 14. `utils.py`：可追溯性和恢复

### L15–16 文本哈希

SHA-256 对 UTF-8 bytes 计算。文件路径相同但内容变化时 hash 会变，resume 因此拒绝把不同
语料接到同一实验。

### L19–40 JSON/CSV

JSON 保存运行/模型/词表；CSV 每次 evaluation 追加一行。`history_last_step` 在恢复前检查
最后记录 step，防止 checkpoint 与日志错位。

### L43–71 checkpoint

字段包括 model/optimizer、step、best val loss、selection metric、运行 config、模型 config、
characters、corpus hash 和 RNG。临时文件写完再 replace，降低进程在保存中断时留下半个
checkpoint 的概率。

`weights_only=True` 不执行任意 pickle 类构造，只加载张量和简单容器，适合本项目保存格式。

## 15. `prepare_data.py` L12–32

DATA_URL 指向 Karpathy 官方 char-rnn 的 raw Tiny Shakespeare。若本地文件存在且未指定
overwrite，只计算字符数/哈希并返回；否则创建父目录、带 timeout 下载 bytes、UTF-8 解码、
写入并报告 source/hash。

真实实验 README 固定记录 1,115,394 字符和 SHA-256，避免“同名数据”不可追踪。

## 16. `train.py` L28–84：参数与安全边界

参数分为数据/输出、训练步数与评估频率、batch/context、模型结构、优化器和设备/状态。
`max_steps` 是更新次数，不是 epoch。`eval_interval` 决定日志/checkpoint 频率，`eval_iters`
决定 train/val loss 各平均多少随机 batch。

`_validate` 拒绝非正步数/interval/batch/block 和非法 optimizer 参数。`_prepare_run` 与 W3/W4
同样拒绝默认覆盖；resume 不清理目录。

## 17. `estimate_loss` L87–99

装饰器关闭 autograd，函数内 `model.eval()` 关闭 dropout。对 train/val 各随机采样
eval_iters 个 batch，把 loss 搬到 CPU 数组后求均值，最后恢复 `model.train()`。

这里的 train loss 是“当前模型在随机 train contexts 上的 eval-mode loss”，不等于最后一个
含 dropout 的训练 batch loss。这样与 validation 更可比。

评估也消耗随机数；checkpoint 保存的是评估和保存之后的 RNG，resume 才能接上下一次训练
batch。

## 18. `train.main` L102–225 逐段精读

### L103–129：语料、split、模型 config

设 seed/device，读语料并编码切分；validation 必须长于 block_size。构造 GPTConfig 后确定
run_dir，创建模型和 AdamW。参数量只统计唯一 Parameter；weight tying 不重复计数。

### L130–167：resume

先验证 eval/log interval、batch、learning rate、weight decay、seed/deterministic，再验证
模型 config、字符列表和 corpus hash，最后加载 model/optimizer。checkpoint 的 step 是
“已经完成的 optimizer updates 且已记录 evaluation 的数量”。恢复循环从同一 step 开始，
跳过已记录 evaluation，然后先完成下一次 update；若错误从 step+1 开始会静默漏掉一次更新。

### L168–183：config 和启动日志

config.json 同时存 CLI、resolved device、model config、characters、hash、总字符数。日志打印
train/val token、vocab 和参数量，先检查这些再等长训练。

### L184–223：训练循环

step 0 先测随机模型；每逢 eval interval 测 loss、追加 CSV、更新 best、保存 last/best。若
不是最后 step，再采 train batch、forward loss、zero grad、backward、gradient clipping、
AdamW step。

gradient clipping：

$$
g\leftarrow g\cdot\min\left(1,\frac{1}{\lVert g\rVert_2}\right).
$$

它限制全局梯度范数，防止偶发大梯度，不替代合适学习率。

### L224–225：画图与结束

训练完成从 CSV 画图并打印 best validation loss。test perplexity 没有单独 split，本项目只用
train/validation；不要把生成片段的主观观感冒充量化 test。

## 19. `generate.py` L17–50

解析 checkpoint、prompt、采样参数与 output。设 seed 让 multinomial 可复现；从 checkpoint
恢复 tokenizer/model config/state；`model.eval()`；把 prompt 包成 batch size 1 的 long tensor；
调用 generate；decode；可选写文件并打印。

prompt 中任何未见字符都会明确报错。PowerShell 换行可用 `` `n ``，实际传入字符必须在 W5
词表中。

## 20. 测试逐项解释

### tokenizer/batch

检查 round trip、shape，并断言 `inputs[:,1:] == targets[:,:-1]`。它直接验证右移关系，而
不是只检查 dtype。

### forward/loss

小 config 保持 n_embd 可被 head 整除，检查 logits `(3,12,7)` 与有限 loss，能抓维度/flatten
错误。

### causal property

两条序列前 5 个 token 相同，后 3 个不同。正确 causal model 的前 5 个位置 logits 必须相同。
这个行为测试比“mask 是下三角 shape”更强：即使 mask shape 看似正确但广播/方向反了也会失败。

### CLI/resume/generate

临时 corpus 和极小模型训练 4 steps，再恢复到 6。CSV `[0,2,4,6]` 证明没有重复/跳步，随后
从 best 独立生成，覆盖序列化和模型重建。

## 21. 参数量怎么估算

每个 block 粗略主要参数：

- QKV：$C\times3C+3C$；
- attention projection：$C\times C+C$；
- MLP：$C\times4C+4C+4C\times C+C$；
- 两个 LayerNorm：$4C$。

所以每 block 主导项约 $12C^2$。再加 token/position embedding、final norm 和 tied head。层数
线性放大参数；context length主要放大 position embedding 与 attention 计算/显存，不直接
放大每层投影权重。

attention score 矩阵是 `(B,H,T,T)`，时间/内存对 T 近似二次：$O(BT^2C)$。这就是长 context
昂贵的核心来源。

## 22. 常见错误

### mask 方向反了

跑 causal property test；打印 4x4 mask，query 行只能看左侧含自身。

### softmax 维度错

每个 query 对所有 keys 的权重和应为 1，所以 `weights.sum(dim=-1)` 近似全 1。

### 忘记 contiguous

transpose 后直接 view 可能报错或重解释错误；先 contiguous 再 view。

### targets dtype/shape 错

必须 long `(B,T)`，CrossEntropy 前 flatten 为 `(B*T)`，不要 one-hot。

### 生成时 dropout 仍开

checkpoint 加载后调用 `model.eval()`；inference_mode 不会自动关闭 dropout。

### resume CSV 重复 step

checkpoint step 代表已记录 evaluation 的已完成更新数。恢复时跳过该 step 的重复 evaluation，
但不能跳过下一次 optimizer update。

## 23. 闭卷自测

1. 为什么 input/target 只相差一个字符？
2. 写出 `(B,T,C)` 到 `(B,H,T,d_h)` 的 view/transpose。
3. 为什么除以 $\sqrt{d_h}$？
4. mask 为什么在 softmax 前填 $-\infty$？
5. attention 和 MLP 分别混合哪个维度的信息？
6. pre-LN block 的两行 forward 是什么？
7. token/position embedding 各表达什么？
8. weight tying 减少多少参数？
9. 为什么生成只取最后位置 logits？
10. temperature/top-k 各怎样改变采样？
11. model.eval 和 inference_mode 各改变什么？
12. causal property test 为什么比检查 mask tensor 更有说服力？
13. 从 `python -m ...train` 口述到第一个 AdamW step。
14. checkpoint 为什么保存 characters 和 corpus hash？
15. 当前实现为什么没有 KV cache，代价是什么？

## 24. 最终验收

- [ ] tokenizer round trip 与 batch 右移关系能手写验证。
- [ ] 不看代码写出 causal attention shape 流。
- [ ] 解释两个 residual、两个 LayerNorm 和 MLP 4C。
- [ ] 4 项 smoke tests 全绿并知道各自边界。
- [ ] Tiny Shakespeare hash 与 config 被记录。
- [ ] 完成正式训练，history/curves/best/last 齐全。
- [ ] 用固定 prompt/temperature/top-k/seed 生成并保存样例。
- [ ] 能解释字符级模型、短 context、无 KV cache 和单语料的局限。

做到最后一项，比生成“像莎士比亚”的几段文本更重要：作品价值来自你能精确说明模型为何
这样运行、证据支持什么、又不能支持什么。

## 25. 2026-08-16 正式实验记录

固定 Tiny Shakespeare SHA、4 layers、4 heads、128 embedding、128 context、batch 64、
AdamW lr $3\times10^{-4}$、weight decay 0.1、seed 42、deterministic，在 RTX 5060 Laptop GPU
上训练 5000 updates：

| 指标 | step 0 | step 5000 / best |
|---|---:|---:|
| train loss | 4.1789 | 1.3960 |
| validation loss | 4.1783 | **1.5749** |
| elapsed | 4.8 s（含首次评估） | 270.7 s |

validation 全程总体下降，step 4250 为 1.6016、step 4500 短暂回到 1.6049、step 4750 再降到
1.5849，最终 step 5000 最好。随机 batch 估计有噪声，所以不能要求每次 evaluation 严格单调。
训练末尾 train/validation 相差约 0.179，说明已出现一定 generalization gap，但 validation 尚未
反弹，当前 5000 steps 内没有证据表明应提前停止。

固定 `ROMEO:` 加换行、temperature 0.8、top-k 40、seed 42 的生成已经出现角色标签、台词换行
和标点结构，也出现 `moder`、`unher`、`followern` 等伪词与语义断裂。字符模型优化的是
next-character likelihood；loss 下降与风格相似不能推出事实正确、长程一致或真正理解。

Windows 下 `conda run` 不能接受包含真实换行的单个参数；生成时应先 `conda activate
ai-learn`，再直接运行 `python -m project4_nanogpt.generate ... --prompt "ROMEO:`n"`。这是
命令包装器限制，不是 tokenizer 或模型错误。
