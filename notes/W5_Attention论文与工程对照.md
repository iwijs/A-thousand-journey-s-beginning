# W5 论文笔记：Transformer 核心怎样落到字符 GPT

论文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)，Vaswani 等，2017。

## 1. 原论文主张与本项目范围

原论文以机器翻译 encoder-decoder 为主要任务，用 attention 取代 recurrent/convolutional
sequence layers，提高训练并行性。本工程只实现 GPT 式 decoder-only、causal language model：
没有 encoder、cross-attention 或翻译 BLEU 实验。

保留的结构核心是 scaled dot-product attention、multi-head、position information、token-wise
feed-forward、residual 和 normalization。

## 2. Scaled dot-product attention

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V.
$$

`models.py/CausalSelfAttention` 中：

- `qkv` 一次投影后 split；
- reshape 为 `(B,H,T,d_h)`；
- $QK^\top$ 得 `(B,H,T,T)`；
- causal mask 将未来位置填 $-\infty$；
- softmax 后对 V 加权；
- heads 拼接并 projection 回 C。

$\sqrt{d_h}$ 缩放用于控制点积方差，避免维度增大使 softmax 过度尖锐。

## 3. 多头的含义

把 C 划成 H 个子空间，不是把同一个 attention matrix 重复 H 次。每头有独立 Q/K/V 切片，
可学习不同的内容匹配模式；拼接后 projection 重新混合各头信息。

## 4. 顺序与位置

self-attention 本身对 token 排列没有足够的位置区分，本工程学习绝对 position embedding，并与
token embedding 相加。它比原论文正弦位置编码更简单，但 context 上限固定为 block size。

## 5. 与原论文实现的差异

- decoder-only causal LM，而非 encoder-decoder translation；
- learned absolute position，而非原论文正弦位置；
- pre-LN，而原始 Transformer 图示常按 post-LN 描述；
- GELU，而原论文 FFN 使用 ReLU；
- 字符 tokenizer、短 context、小语料；
- 没有 KV cache、distributed training 或现代 fused kernels。

这些差异必须写清，否则“从零复现 Transformer”容易被误解为逐配置复现原论文。

## 6. 工程证据

- causal behavior test 修改未来 token，早期 logits 保持相同；
- input/target 右移测试验证 next-token supervision；
- checkpoint 保存 model config、字符表和 corpus hash；
- train/validation loss 曲线而非只给一段生成文本；
- 固定 prompt/temperature/top-k/seed 便于样例比较。

## 7. 2026-08-16 实际结果

818,048 参数模型在 Tiny Shakespeare 上训练 5000 updates，validation loss 从 4.1783 降到
1.5749，最佳点为最后的 step 5000。固定采样已经产生角色标签、换行和类似台词的局部形式，
同时仍有伪词和语义断裂。它支持“causal Transformer 学到了字符语料的统计结构”，不支持
“复现了原论文翻译结果”或“模型具有语言理解”的结论。

## 8. 我能否回答

1. 为什么 attention score 对 T 是二次内存？
2. softmax 为什么沿 key-position 维？
3. causal mask 为什么在 softmax 前加？
4. attention 与 FFN 分别混合 token/feature 哪一维？
5. 当前工程与原论文至少五个差异是什么？
