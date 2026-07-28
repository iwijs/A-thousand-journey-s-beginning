# -*- coding: utf-8 -*-
"""
W1 Day4：张量变形与自动求导
============================
运行自动判分：
    conda run --no-capture-output -n ai-learn python day4_autograd_practice.py

练习重点：reshape/flatten、permute、批量矩阵乘法、autograd、手写梯度下降。
不要修改自动判分部分，也不要使用 nn.Module 或 torch.optim 完成第 6 题。
"""

import torch


def flatten_images(images):
    """题1：保留 batch 维，将其余图像维度展平。

    输入 shape (B, C, H, W)，返回 shape (B, C*H*W)。
    """
    return images.flatten(start_dim=1)


def hwc_to_chw(image):
    """题2：把单张图像从 (H, W, C) 调整为 (C, H, W)。"""
    return image.permute(2, 0, 1)


def pairwise_scores(query, key):
    """题3：批量计算两组向量的两两点积分数。

    query: (B, N, D)
    key:   (B, M, D)
    返回:  (B, N, M)
    """
    return query @ key.transpose(-2, -1)


def scalar_value_and_grad(x_value):
    """题4：用 autograd 计算 y=x^3+2x^2-x+1 的函数值和 dy/dx。

    返回两个 Python float：(y_value, gradient)。
    不允许手写导数公式。
    """
    x = torch.tensor(float(x_value), dtype=torch.float32, requires_grad=True)
    y = x**3 + 2 * x**2 - x + 1
    y.backward()
    return y.item(), x.grad.item()


def mse_parameter_grads(x, y, w_value, b_value):
    """题5：计算线性模型的均方误差及其对 w、b 的梯度。

    pred = w * x + b
    loss = mean((pred - y)^2)
    返回三个 Python float：(loss, dloss_dw, dloss_db)。
    不允许手写梯度公式。
    """
    w = torch.tensor(
        w_value, dtype=x.dtype, device=x.device, requires_grad=True
    )
    b = torch.tensor(
        b_value, dtype=x.dtype, device=x.device, requires_grad=True
    )
    predictions = w * x + b
    loss = ((predictions - y) ** 2).mean()
    loss.backward()
    return loss.item(), w.grad.item(), b.grad.item()


def train_linear_regression(x, y, steps=200, learning_rate=0.1):
    """题6：不用 nn.Module 和 torch.optim，手写梯度下降拟合 y=w*x+b。

    初始 w=b=0。每轮必须包含：forward、loss、backward、no_grad 更新、清空梯度。
    返回三个 Python float：(w, b, final_loss)。
    final_loss 应按更新后的最终参数重新计算。
    """
    w = torch.zeros((), dtype=x.dtype, device=x.device, requires_grad=True)
    b = torch.zeros((), dtype=x.dtype, device=x.device, requires_grad=True)

    for _ in range(steps):
        predictions = w * x + b
        loss = ((predictions - y) ** 2).mean()
        loss.backward()

        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad

        w.grad.zero_()
        b.grad.zero_()

    with torch.no_grad():
        final_loss = (((w * x + b) - y) ** 2).mean()

    return w.item(), b.item(), final_loss.item()


# ========== 自动判分（不要修改） ==========

def _check_scalar_grad():
    value, grad = scalar_value_and_grad(2.0)
    return abs(value - 15.0) < 1e-6 and abs(grad - 19.0) < 1e-6


def _check_mse_grad():
    # x=[1,2], y=[3,5], w=1, b=0
    # pred=[1,2], error=[-2,-3], loss=(4+9)/2=6.5
    # dL/dw = mean(2*error*x) = (-4-12)/2 = -8
    # dL/db = mean(2*error)   = (-4-6)/2  = -5
    loss, dw, db = mse_parameter_grads(
        torch.tensor([1.0, 2.0]),
        torch.tensor([3.0, 5.0]),
        1.0,
        0.0,
    )
    return abs(loss - 6.5) < 1e-6 and abs(dw + 8.0) < 1e-6 and abs(db + 5.0) < 1e-6


def _check_training():
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = 2.0 * x + 1.0
    w, b, final_loss = train_linear_regression(x, y)
    return abs(w - 2.0) < 1e-3 and abs(b - 1.0) < 1e-3 and final_loss < 1e-6


def _check():
    images = torch.arange(2 * 3 * 2 * 2).reshape(2, 3, 2, 2)
    image_hwc = torch.arange(2 * 3 * 2).reshape(2, 3, 2)

    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])      # (1, 2, 2)
    key = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [-1.0, 1.0]]])  # (1, 3, 2)
    expected_scores = torch.tensor([[[1.0, 3.0, -1.0], [2.0, 4.0, 1.0]]])

    tests = [
        ("题1 保留batch展平", lambda: torch.equal(
            flatten_images(images), images.reshape(2, 12)
        )),
        ("题2 HWC转CHW", lambda: torch.equal(
            hwc_to_chw(image_hwc), image_hwc.permute(2, 0, 1)
        ) and tuple(hwc_to_chw(image_hwc).shape) == (2, 2, 3)),
        ("题3 批量两两点积", lambda: torch.allclose(
            pairwise_scores(query, key), expected_scores
        )),
        ("题4 标量autograd", _check_scalar_grad),
        ("题5 MSE参数梯度", _check_mse_grad),
        ("题6 手写梯度下降", _check_training),
    ]

    print("\nDay4 变形与 Autograd 练习自动判分")
    print("=" * 50)
    passed = 0
    for name, test in tests:
        try:
            ok = bool(test())
        except Exception as exc:
            ok = False
            name += f"  (报错: {type(exc).__name__}: {exc})"
        print("  " + ("[OK]" if ok else "[--]") + "  " + name)
        passed += ok
    print("=" * 50)
    if passed == len(tests):
        print(f"通过 {passed}/{len(tests)}  全部完成！你已经跑通最小训练闭环。")
    else:
        print(f"通过 {passed}/{len(tests)}  按手册中的调试顺序排查。")


if __name__ == "__main__":
    _check()
