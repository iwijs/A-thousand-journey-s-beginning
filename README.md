# A-thousand-journey-s-beginning

我的 AI 学习与作品集仓库 · 从零开始的科研之路。

> 目标：暑假 7 周打穿 PyTorch + AI 底层原理（工程实战优先），沉淀 4–5 个可展示项目，
> 为联系导师、进入科研组做准备。学习方法论参考高宁《Zero-to-Hero 成为科研高手》。

## 当前状态

- W1 Day2 已完成并提交。
- W1 Day3–Day6 练习脚本已完成并验证，自动判分依次为 8/8、6/6、7/7、6/6。
- `project0_micrograd` 已有可运行参考实现和测试。
- `project1_mnist` 已通过 synthetic 离线端到端验证；正式 MNIST 训练结果待本人运行和记录。

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
├── project3_resnet/         # 项目③ ResNet 复现
├── project4_nanogpt/        # 项目④ 字符级 GPT
├── project5_frontier/       # 项目⑤ 前沿方向小实验
└── notes/                   # 学习笔记 / 论文精读
```

规划中的目录会在进入相应周次时创建；当前不要把尚未创建的 W3–W6 项目视为已完成。

## 环境与运行方式

本机当前系统默认 `python` 指向 `C:\Python314\python.exe`，其中没有安装 NumPy/PyTorch。
已验证的学习环境是 `ai-learn`（Python 3.11、NumPy 2.4.4、PyTorch 2.11.0+cpu）。
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

## 进度日志

- 2026-07-06：W1 Day1，环境搭建完成，跑通 `day1_env_check.py`。
- 2026-07-18：W1 Day2，完成 Python 与 NumPy 练习并提交。
- 2026-07-28：搭建 W2 micrograd/MNIST 参考工程；离线测试通过，正式实验待完成。
- 2026-07-28：完成 W1 Day3–Day6 练习；在 `ai-learn` 环境逐个自动判分，结果为 8/8、6/6、7/7、6/6。

## 关于我
人工智能专业本科生（浙江大学）。正在系统学习深度学习，寻找感兴趣的研究方向。
