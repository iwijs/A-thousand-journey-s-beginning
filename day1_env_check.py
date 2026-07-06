#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
day1_env_check.py —— W1 第一天环境自检 / 依赖清点脚本

作用：在你的 conda 环境里跑一次，它会：
  1) 报告当前 Python 与所在环境路径
  2) 逐个检查 Day1 需要的包：已装(带版本) / 缺失
  3) 若 PyTorch 已装，跑一段"一步反向传播"demo（W2 会深入的 autograd）
  4) 末尾给出一条"把缺的一次装齐"的命令

用法：
    conda env list                 # 先找到你那个 AI 学习环境的名字
    conda activate <环境名>         # 例如 conda activate ai
    cd Desktop\AI\2_成果作品\A-thousand-journey-s-beginning
    python day1_env_check.py

把输出发给 Claude，我帮你确认还差什么。
"""

import sys
import platform
import importlib.util
from importlib import import_module

# (显示名, 导入名, pip 包名, 说明) —— Day1 所需
REQUIRED = [
    ("NumPy",        "numpy",        "numpy",        "数值计算基础"),
    ("PyTorch",      "torch",        "torch",        "本暑假主线"),
    ("TorchVision",  "torchvision",  "torchvision",  "数据集/图像变换（MNIST/CIFAR 要用）"),
    ("Matplotlib",   "matplotlib",   "matplotlib",   "画图/看训练曲线"),
    ("Jupyter",      "jupyter_core", "jupyter",      "notebook 交互（写实验）"),
    ("Playwright",   "playwright",   "playwright",   "导师爬虫用（真实浏览器）"),
    ("openpyxl",     "openpyxl",     "openpyxl",     "导师爬虫导出 Excel"),
]


def get_version(import_name):
    try:
        mod = import_module(import_name)
        return getattr(mod, "__version__", "已装")
    except Exception:
        try:
            from importlib.metadata import version
            return version(import_name.replace("_", "-"))
        except Exception:
            return "已装"


def check_packages():
    print("=" * 58)
    print("依赖清点（Day1 所需）")
    print("-" * 58)
    missing = []
    for disp, imp, pip_name, note in REQUIRED:
        found = importlib.util.find_spec(imp) is not None
        if found:
            print("  [OK]   %-12s %-16s %s" % (disp, get_version(imp), note))
        else:
            print("  [缺失] %-12s %-16s %s" % (disp, "-", note))
            missing.append(pip_name)
    return missing


def torch_demo():
    if importlib.util.find_spec("torch") is None:
        return
    import torch
    print("=" * 58)
    print("PyTorch 快速自测")
    has_cuda = torch.cuda.is_available()
    gpu = ("有：" + torch.cuda.get_device_name(0)) if has_cuda else "否（本地 CPU 就够，CIFAR 阶段再上 AutoDL）"
    print("  GPU 可用 : " + gpu)
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    print("  张量测试 : x.sum()=%.1f, x.mean()=%.2f" % (x.sum().item(), x.mean().item()))
    print("-" * 58)
    print("  一步反向传播 demo（W2 会手写 micrograd 深入）：")
    w = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    x0, y = torch.tensor(3.0), torch.tensor(10.0)
    pred = w * x0 + b
    loss = (pred - y) ** 2
    loss.backward()
    print("     pred=w*x+b=%.1f, loss=(pred-y)^2=%.1f" % (pred.item(), loss.item()))
    print("     dloss/dw=%.1f (应为 -18)   dloss/db=%.1f (应为 -6)" % (w.grad.item(), b.grad.item()))
    ok = abs(w.grad.item() + 18) < 1e-4 and abs(b.grad.item() + 6) < 1e-4
    print("  PyTorch + autograd 正常" if ok else "  梯度异常，把输出发 Claude")


def playwright_hint():
    if importlib.util.find_spec("playwright") is None:
        return
    print("=" * 58)
    print("Playwright 提示：包已装，但还需下载浏览器内核（只需一次）：")
    print("     playwright install chromium")
    print("  （已装过就忽略。爬虫在 3_工具 里的 zju_advisor_crawler）")


def main():
    print("")
    print("W1 Day1 环境自检")
    print("=" * 58)
    print("① 当前环境")
    print("  Python 版本 : " + platform.python_version())
    print("  解释器路径 : " + sys.executable)
    print("  （路径里含 envs 后跟名字，就是你当前激活的 conda 环境）")
    ok_py = sys.version_info >= (3, 9)
    print("  版本 >=3.9  : " + ("OK" if ok_py else "建议 3.10/3.11"))

    missing = check_packages()
    torch_demo()
    playwright_hint()

    print("=" * 58)
    if not missing:
        print("Day1 依赖全部齐活！环境就绪，可以开始学习了。")
        print("下一步：把运行结果截图，commit 到 A-thousand-journey-s-beginning 仓库。")
    else:
        print("还差这些包，复制下面这条命令一次装齐：")
        print("")
        print("   pip install " + " ".join(missing))
        print("")
        print("下载慢就加清华镜像：")
        print("   pip install " + " ".join(missing) + " -i https://pypi.tuna.tsinghua.edu.cn/simple")
        if "playwright" in missing:
            print("")
            print("（playwright 装完还要跑一次：playwright install chromium）")
    print("")


if __name__ == "__main__":
    main()
