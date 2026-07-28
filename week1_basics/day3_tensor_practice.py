# -*- coding: utf-8 -*-
"""
W1 Day3：PyTorch Tensor 基础
============================
运行自动判分：
    conda run --no-capture-output -n ai-learn python day3_tensor_practice.py

练习重点：shape / dtype / device、广播、索引、矩阵乘法、NumPy 互转。
不要修改自动判分部分，也不要把测试答案硬编码进函数。
"""

import numpy as np
import torch


def make_matrix():
    """题1：返回包含 1~6、shape=(2, 3)、dtype=float32 的 Tensor。"""
    return torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)


def tensor_info(x):
    """题2：返回 (shape元组, dtype, device类型字符串)。

    例：CPU 上 float32 的 (2,3) Tensor 应返回：
        ((2, 3), torch.float32, "cpu")
    """
    return tuple(x.shape), x.dtype, x.device.type


def standardize_columns(x):
    """题3：对二维浮点 Tensor 的每一列分别标准化。

    返回 (x - 列均值) / 列标准差。
    标准差必须使用总体标准差，即 torch.std(..., correction=0)。
    要求使用 dim=0 和 keepdim=True，让广播关系清楚可见。
    """
    column_mean = x.mean(dim=0, keepdim=True)
    column_std = x.std(dim=0, keepdim=True, correction=0)
    return (x - column_mean) / column_std


def add_channel_bias(images, bias):
    """题4：给图像批次的每个通道加对应偏置。

    images: shape (B, C, H, W)
    bias:   shape (C,)
    return: shape (B, C, H, W)
    """
    channel_bias = bias.reshape(1, -1, 1, 1)
    return images + channel_bias


def positive_even_elements(x):
    """题5：用布尔索引返回一维整数 Tensor 中所有正偶数，保持原顺序。"""
    mask = (x > 0) & (x % 2 == 0)
    return x[mask]


def batch_linear(x, weight, bias):
    """题6：实现一个批量线性变换 x @ weight + bias。

    x:      (B, in_features)
    weight: (in_features, out_features)
    bias:   (out_features,)
    """
    return x @ weight + bias


def numpy_shared_tensor(array):
    """题7：用 array 创建一个共享底层内存的 CPU Tensor。

    不允许复制数据；保持原 dtype。输入保证是 NumPy ndarray。
    """
    return torch.from_numpy(array)


def to_best_device(x):
    """题8：把 x 移到当前可用的最佳设备并返回。

    有 CUDA 时用 cuda，否则用 cpu。不要直接写死 x.cuda()。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return x.to(device)


# ========== 自动判分（不要修改） ==========

def _is_close(x, expected, atol=1e-6):
    return isinstance(x, torch.Tensor) and torch.allclose(
        x.detach().cpu(), expected.detach().cpu(), atol=atol, rtol=1e-5
    )


def _check_numpy_sharing():
    array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    tensor = numpy_shared_tensor(array)
    if not isinstance(tensor, torch.Tensor):
        return False
    array[1] = 99.0
    return tensor.device.type == "cpu" and tensor.dtype == torch.float32 and tensor[1].item() == 99.0


def _check():
    column_input = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    standardized = torch.tensor([
        [-1.2247449, -1.2247449],
        [0.0, 0.0],
        [1.2247449, 1.2247449],
    ])

    images = torch.zeros(2, 3, 2, 2)
    bias = torch.tensor([1.0, 2.0, 3.0])
    expected_images = bias.reshape(1, 3, 1, 1).expand(2, 3, 2, 2)

    linear_x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    linear_w = torch.tensor([[1.0, 0.0, -1.0], [0.5, 2.0, 1.0]])
    linear_b = torch.tensor([0.5, -0.5, 1.0])
    expected_linear = torch.tensor([[2.5, 3.5, 2.0], [5.5, 7.5, 2.0]])

    expected_device = "cuda" if torch.cuda.is_available() else "cpu"

    tests = [
        ("题1 创建Tensor", lambda: _is_close(
            make_matrix(), torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        ) and make_matrix().dtype == torch.float32),
        ("题2 shape/dtype/device", lambda: tensor_info(torch.zeros(2, 3)) == (
            (2, 3), torch.float32, "cpu"
        )),
        ("题3 按列标准化", lambda: _is_close(
            standardize_columns(column_input), standardized, atol=1e-5
        )),
        ("题4 通道偏置广播", lambda: _is_close(
            add_channel_bias(images, bias), expected_images
        )),
        ("题5 布尔索引", lambda: torch.equal(
            positive_even_elements(torch.tensor([-4, -1, 0, 2, 3, 6, 7])),
            torch.tensor([2, 6]),
        )),
        ("题6 批量线性变换", lambda: _is_close(
            batch_linear(linear_x, linear_w, linear_b), expected_linear
        )),
        ("题7 NumPy共享内存", _check_numpy_sharing),
        ("题8 最佳device", lambda: to_best_device(torch.ones(2)).device.type == expected_device),
    ]

    print("\nDay3 Tensor 练习自动判分")
    print("=" * 46)
    passed = 0
    for name, test in tests:
        try:
            ok = bool(test())
        except Exception as exc:
            ok = False
            name += f"  (报错: {type(exc).__name__}: {exc})"
        print("  " + ("[OK]" if ok else "[--]") + "  " + name)
        passed += ok
    print("=" * 46)
    if passed == len(tests):
        print(f"通过 {passed}/{len(tests)}  全部完成！先复盘，再 commit。")
    else:
        print(f"通过 {passed}/{len(tests)}  先打印 shape/dtype/device，再定位未通过题目。")


if __name__ == "__main__":
    _check()
