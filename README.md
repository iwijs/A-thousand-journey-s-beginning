# A-thousand-journey-s-beginning

我的 AI 学习与作品集仓库 · 从零开始的科研之路。

> 目标：暑假 7 周打穿 PyTorch + AI 底层原理（工程实战优先），沉淀 4–5 个可展示项目，
> 为联系导师、进入科研组做准备。学习方法论参考高宁《Zero-to-Hero 成为科研高手》。

## 当前状态

- W1 Day2 已完成并提交。
- W1 Day3–Day6 练习脚本已完成并验证，自动判分依次为 8/8、6/6、7/7、6/6。
- `project0_micrograd` 已完成学习实操与扩展：5 项测试通过，完成 `sigmoid`、有限差分检查
  和三组学习率对比。
- `project1_mnist` 已完成正式 MNIST 训练、独立 checkpoint 评估、曲线与单变量对照实验。
- `project2_cifar10` 已完成三档增强、CPU/GPU 统一训练、CSV、完整断点恢复和 6 项测试；
  30 轮真实 CIFAR-10 baseline 可确定性复现，best validation / 独立 test 为
  88.42% / 87.04%。只关闭数据增强的一变量对照为 85.92% / 84.36%，并呈现明显更大的
  train-validation gap；test 仅在 validation 完成选模后做最终评估。
- `project3_resnet` 已手写 CIFAR ResNet-18/plain18/BN 消融并通过 4 项测试；100 轮 ResNet-18
  / Plain-18 的独立 test 为 94.64% / 94.19%。
- `project4_nanogpt` 已实现字符 tokenizer、causal multi-head attention、GPT 训练/恢复/生成
  并通过 4 项测试；818,048 参数模型训练 5000 steps，最佳 validation loss 1.5749。
- `project5_lora` 已实现低秩注入、冻结、merge、adapter checkpoint 与领域语料抽取并通过
  5 项测试；rank 2/8 最佳 domain validation loss 为 1.4294 / 1.4299。
- `notes/` 已加入 ResNet、Attention、LoRA 三篇“原论文主张 ↔ 当前实现 ↔ 证据边界”对照笔记。

参考实现和自动测试只是学习起点，不把它们写成已经独立完成的个人成果。

## 技术栈

Python · NumPy · PyTorch · torchvision · matplotlib · Git/GitHub

## 目录规划
```
A-thousand-journey-s-beginning/
├── README.md
├── day1_env_check.py        # 环境自检脚本
├── week1_basics/            # W1 Python/NumPy/张量练习
├── project0_micrograd/      # 标量自动求导 + 小型 MLP
├── project1_mnist/          # 项目① MNIST（MLP + CNN）
├── project2_cifar10/        # 项目② CIFAR-10 训练工程
├── project3_resnet/         # 项目③ 手写 ResNet-CIFAR 与消融
├── project4_nanogpt/        # 项目④ 从零字符级 GPT
├── project5_lora/           # 项目⑤ LoRA 前沿小实验
├── docs/tutorials/          # W3–W6 理论结合工程逐行教程
└── notes/                   # 学习笔记 / 论文精读
```

项目目录和 smoke tests 已建立，但正式实验只以各项目 README 中记录的真实日志为准；不要把
“代码存在”或 synthetic 通过写成正式数据集成果。

## 环境与运行方式

本机当前系统默认 `python` 指向 `C:\Python314\python.exe`，其中没有安装 NumPy/PyTorch。
已验证的学习环境是 `ai-learn`（Python 3.11.15、NumPy 2.4.4、PyTorch 2.11.0+cu130，
本机 NVIDIA GeForce RTX 5060 Laptop GPU）。
为避免误用系统 Python，也避免 Conda 捕获中文输出时的编码错误，推荐从仓库根目录直接运行：

```powershell
conda run --no-capture-output -n ai-learn python -c "import sys, numpy, torch; print(sys.executable); print(numpy.__version__); print(torch.__version__)"
conda run --no-capture-output -n ai-learn python week1_basics/day3_tensor_practice.py
conda run --no-capture-output -n ai-learn python week1_basics/day4_autograd_practice.py
conda run --no-capture-output -n ai-learn python week1_basics/day5_nn_training_practice.py
conda run --no-capture-output -n ai-learn python week1_basics/day6_data_pipeline_practice.py
```

交互学习时也可以先执行 `conda activate ai-learn`，确认 `python -c "import sys; print(sys.executable)"`
输出路径包含 `envs\ai-learn` 后，再使用普通 `python` 命令。

W2 工程线的完整操作讲义位于工作区：

- `1_每日任务指南/W2_工程实践01_micrograd完整教程.md`
- `1_每日任务指南/W2_工程实践02_MNIST完整教程.md`

W3–W6 的可发布逐行工程讲义位于 [`docs/tutorials`](docs/tutorials/README.md)；本地完整理论
索引仍是 `1_每日任务指南/W2-W6_理论讲义索引.md`。每个正式实验的真实结果、曲线、配置和
证据边界已回填到对应项目 README。

## 进度日志

- 2026-07-06：W1 Day1，环境搭建完成，跑通 `day1_env_check.py`。
- 2026-07-18：W1 Day2，完成 Python 与 NumPy 练习并提交。
- 2026-07-28：搭建 W2 micrograd/MNIST 参考工程；离线测试通过，正式实验待完成。
- 2026-07-28：完成 W1 Day3–Day6 练习；在 `ai-learn` 环境逐个自动判分，结果为 8/8、6/6、7/7、6/6。
- 2026-07-30：完成 W2 micrograd 学习实操；实现 `sigmoid`，加入有限差分检查，5 项测试
  通过，并记录 `0.01`、`0.08`、`0.1` 三组学习率实验。
- 2026-08-10：搭建 W3 CIFAR-10 基础 CNN 训练工程；实现 train/validation 固定划分、
  validation-only best checkpoint、last checkpoint、训练增强、CSV、曲线和离线端到端测试。
- 2026-08-16：完成 W3 三档增强与断点恢复；RTX 5060 Laptop GPU 上完成 30 轮真实 baseline，
  best validation 88.42%，一次独立官方 test 87.04%。
- 2026-08-17：校验 CIFAR-10 官方归档 MD5 并绕开旧解压目录 ACL 异常，在新数据根目录
  精确复现 baseline；随后仅把 `augmentation` 从 `basic` 改为 `none` 跑满同样 30 轮。
  无增强对照 best validation / final test 为 85.92% / 84.36%，分别低 2.50 / 2.68 pp，
  同时选中轮 train-validation gap 从 2.53 pp 扩大到 13.76 pp。完整命令、配置、曲线和
  输出位于 `project2_cifar10/README.md` 与 `assets/`。
- 2026-08-16：完成 W4 ResNet-18 / Plain-18 两组 100 轮对照，official test 分别为
  94.64% / 94.19%；差距很小，但 Plain 高学习率阶段波动更大。
- 2026-08-16：完成 W5 Tiny Shakespeare 字符 GPT 5000-step baseline，validation loss
  4.1783 → 1.5749，并保存固定生成样例。
- 2026-08-16：完成 W6 attention LoRA rank 2/8 对照；修复正式 CUDA 首跑发现的 adapter
  device/dtype bug，扩展为 5 项测试，并如实记录“r8 train 更低但 validation 无收益”。

## 关于我
人工智能专业本科生（浙江大学）。正在系统学习深度学习，寻找感兴趣的研究方向。
