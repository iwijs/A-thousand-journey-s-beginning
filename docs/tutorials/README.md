# W3–W6 理论结合工程逐行教程

这些教程对应仓库中的四个连续工程，目标不是复述 API，而是把动机、数学、tensor shape、
源码调用链、训练状态、测试证据和真实实验结果连成一条线。

## 阅读顺序

1. [`W3_CNN与训练工程化_逐行教程.md`](W3_CNN与训练工程化_逐行教程.md)：卷积、数据协议、
   CPU/CUDA、CSV、checkpoint 与完整断点恢复；
2. [`W4_ResNet_逐行教程.md`](W4_ResNet_逐行教程.md)：BasicBlock、projection、残差梯度、
   SGD/cosine 和 Plain-18 受控对照；
3. [`W5_字符级GPT_逐行教程.md`](W5_字符级GPT_逐行教程.md)：字符 token、Q/K/V、causal mask、
   pre-LN Transformer、训练与 autoregressive generation；
4. [`W6_LoRA_逐行教程.md`](W6_LoRA_逐行教程.md)：低秩更新、冻结、递归注入、merge、adapter
   checkpoint，以及 rank 2/8 对照。

## 学习方式

- 第一次只画调用链和 shape，不追求记住所有行；
- 第二次遮住源码，重写关键函数，再用项目 tests 验证；
- 第三次修改一个变量，先写预测，再看 CSV/曲线；
- 最后用“问题、方法、关键实现、真实结果、局限”五句话介绍每个项目。

教程和参考实现由 AI 协助建立，不等于学习者已经独立掌握。作品展示时应准确描述自己实际
完成的阅读、重写、修改和诊断，不能把文件存在直接当成个人能力证明。
