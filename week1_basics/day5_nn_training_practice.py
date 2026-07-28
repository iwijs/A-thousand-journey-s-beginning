# -*- coding: utf-8 -*-
"""
W1 Day5：nn.Module 与标准训练闭环
================================
运行自动判分：
    conda run --no-capture-output -n ai-learn python day5_nn_training_practice.py

练习重点：nn.Module、nn.Linear、Sequential、loss、optimizer、state_dict。
不要修改自动判分部分，也不要把测试答案硬编码进函数。
"""

import torch
from torch import nn


class LinearRegressionModel(nn.Module):
    """题1：用一个 nn.Linear 实现一元线性回归。

    输入 shape (B, 1)，输出 shape (B, 1)。
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def linear_from_parameters(x, weight, bias):
    """题2：按 nn.Linear 的参数布局手动计算前向。

    x:      (B, in_features)
    weight: (out_features, in_features)
    bias:   (out_features,)
    返回:   (B, out_features)

    注意：nn.Linear.weight 的布局与 Day3 手写 weight 的布局相反。
    """
    return x @ weight.T + bias


def build_mlp(in_features, hidden_features, out_features):
    """题3：用 nn.Sequential 构造两层 MLP。

    结构必须是：
        Linear(in_features, hidden_features)
        ReLU()
        Linear(hidden_features, out_features)
    """
    return nn.Sequential(
        nn.Linear(in_features, hidden_features),
        nn.ReLU(),
        nn.Linear(hidden_features, out_features),
    )


def count_trainable_parameters(model):
    """题4：返回模型中所有可训练标量参数的总数。

    只统计 requires_grad=True 的参数；不得针对某个固定模型硬编码。
    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def classification_loss_and_predictions(logits, labels):
    """题5：计算多分类交叉熵，并给出预测类别。

    logits: (B, C)，未经 softmax 的原始分数
    labels: (B,)，dtype 应为 torch.int64
    返回: (标量 loss Tensor, shape=(B,) 的预测类别 Tensor)
    """
    loss = nn.CrossEntropyLoss()(logits, labels)
    predictions = logits.argmax(dim=1)
    return loss, predictions


def train_linear_regression(x, y, steps=200, learning_rate=0.1):
    """题6：用标准 PyTorch 组件拟合 y = w*x + b。

    必须使用：
    - LinearRegressionModel
    - nn.MSELoss
    - torch.optim.SGD
    - zero_grad -> forward -> loss -> backward -> step

    为便于自动判分，请把线性层的 weight 和 bias 都初始化为 0。
    返回 (训练后的 model, 按最终参数重新计算的 Python float loss)。
    """
    model = LinearRegressionModel()
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.zero_()

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for _ in range(steps):
        optimizer.zero_grad()
        predictions = model(x)
        loss = loss_fn(predictions, y)
        loss.backward()
        optimizer.step()

    with torch.inference_mode():
        final_loss = loss_fn(model(x), y).item()

    return model, final_loss


def copy_model_state(source_model, target_model):
    """题7：把 source_model 的参数和 buffer 复制到 target_model。

    使用 state_dict / load_state_dict，不要逐层手抄参数。
    返回加载后的 target_model。
    """
    target_model.load_state_dict(source_model.state_dict())
    return target_model


# ========== 自动判分（不要修改） ==========

def _check_model_definition():
    model = LinearRegressionModel()
    x = torch.randn(4, 1)
    output = model(x)
    parameters = dict(model.named_parameters())
    return (
        isinstance(model, nn.Module)
        and tuple(output.shape) == (4, 1)
        and set(parameters) == {"linear.weight", "linear.bias"}
        and tuple(parameters["linear.weight"].shape) == (1, 1)
        and tuple(parameters["linear.bias"].shape) == (1,)
    )


def _check_linear_layout():
    x = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
    weight = torch.tensor([
        [1.0, 0.5],
        [-2.0, 1.0],
        [0.0, -1.0],
    ])
    bias = torch.tensor([0.5, 1.0, -0.5])
    expected = torch.tensor([
        [2.5, 1.0, -2.5],
        [1.0, 6.0, -3.5],
    ])
    result = linear_from_parameters(x, weight, bias)
    return isinstance(result, torch.Tensor) and torch.allclose(result, expected)


def _check_mlp():
    model = build_mlp(4, 6, 3)
    if not isinstance(model, nn.Sequential):
        return False
    layers = list(model.children())
    output = model(torch.randn(5, 4))
    return (
        len(layers) == 3
        and isinstance(layers[0], nn.Linear)
        and isinstance(layers[1], nn.ReLU)
        and isinstance(layers[2], nn.Linear)
        and tuple(output.shape) == (5, 3)
    )


def _check_parameter_count():
    # (4*6 + 6) + (6*3 + 3) = 51
    model = build_mlp(4, 6, 3)
    return count_trainable_parameters(model) == 51


def _check_classification():
    logits = torch.tensor([
        [3.0, 1.0, -1.0],
        [0.0, 2.0, 1.0],
        [-1.0, 0.0, 4.0],
    ])
    labels = torch.tensor([0, 2, 2])
    result = classification_loss_and_predictions(logits, labels)
    if not isinstance(result, tuple) or len(result) != 2:
        return False
    loss, predictions = result
    expected_loss = nn.CrossEntropyLoss()(logits, labels)
    return (
        isinstance(loss, torch.Tensor)
        and loss.ndim == 0
        and torch.allclose(loss, expected_loss)
        and torch.equal(predictions, torch.tensor([0, 1, 2]))
    )


def _check_training():
    x = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    y = 2.0 * x + 1.0
    model, final_loss = train_linear_regression(x, y)
    if not isinstance(model, LinearRegressionModel):
        return False
    weight = model.linear.weight.item()
    bias = model.linear.bias.item()
    return (
        isinstance(final_loss, float)
        and abs(weight - 2.0) < 1e-3
        and abs(bias - 1.0) < 1e-3
        and final_loss < 1e-6
    )


def _check_state_copy():
    torch.manual_seed(7)
    source = build_mlp(2, 4, 2)
    torch.manual_seed(99)
    target = build_mlp(2, 4, 2)
    x = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    before = target(x).detach().clone()
    loaded = copy_model_state(source, target)
    after = loaded(x).detach()
    return (
        loaded is target
        and not torch.allclose(before, after)
        and torch.allclose(after, source(x).detach())
    )


def _check():
    tests = [
        ("题1 定义nn.Module", _check_model_definition),
        ("题2 nn.Linear参数布局", _check_linear_layout),
        ("题3 构造两层MLP", _check_mlp),
        ("题4 统计可训练参数", _check_parameter_count),
        ("题5 交叉熵与预测", _check_classification),
        ("题6 标准训练闭环", _check_training),
        ("题7 复制state_dict", _check_state_copy),
    ]

    print("\nDay5 nn.Module 与训练闭环练习自动判分")
    print("=" * 54)
    passed = 0
    for name, test in tests:
        try:
            ok = bool(test())
        except Exception as exc:
            ok = False
            name += f"  (报错: {type(exc).__name__}: {exc})"
        print("  " + ("[OK]" if ok else "[--]") + "  " + name)
        passed += ok
    print("=" * 54)
    if passed == len(tests):
        print(f"通过 {passed}/{len(tests)}  全部完成！你已掌握标准 PyTorch 训练闭环。")
    else:
        print(f"通过 {passed}/{len(tests)}  先检查模型结构、shape 和梯度清零顺序。")


if __name__ == "__main__":
    _check()
