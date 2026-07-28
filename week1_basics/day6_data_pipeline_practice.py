# -*- coding: utf-8 -*-
"""
W1 Day6：Dataset、DataLoader 与训练/验证流程
===========================================
运行自动判分：
    conda run --no-capture-output -n ai-learn python day6_data_pipeline_practice.py

练习重点：Dataset、mini-batch、train/eval、指标聚合、device、state_dict 持久化。
不要修改自动判分部分，也不要把测试答案硬编码进函数。
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


class PointDataset(Dataset):
    """题1：保存二维特征与分类标签的自定义 Dataset。"""

    def __init__(self, features, labels):
        if len(features) != len(labels):
            raise ValueError("features 和 labels 的样本数必须相同")
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


def make_dataloader(features, labels, batch_size, shuffle=False):
    """题2：用 TensorDataset 和 DataLoader 组成 mini-batch。

    不要设置 drop_last=True；最后一个不足 batch_size 的 batch 必须保留。
    """
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class TinyClassifier(nn.Module):
    """供后续题目使用的两层分类 MLP。"""

    def __init__(self, in_features=2, hidden_features=8, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """题3：训练一个 epoch，并返回按样本加权的平均 loss。

    每个 batch 都要：
    1. 把 features、labels 移到 device；
    2. zero_grad -> forward -> loss -> backward -> step；
    3. 用 batch 样本数累计 loss。

    返回 Python float。函数开始时要调用 model.train()。
    """
    model.train()
    total_loss = 0.0
    total_examples = 0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("dataloader 不能为空")
    return total_loss / total_examples


def evaluate(model, dataloader, loss_fn, device):
    """题4：在验证集上返回 (平均 loss, accuracy)。

    要求：
    - 调用 model.eval()；
    - 使用 torch.no_grad() 或 torch.inference_mode()；
    - loss 按样本数加权；
    - accuracy 是 0~1 的 Python float。
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.inference_mode():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = loss_fn(logits, labels)
            predictions = logits.argmax(dim=1)

            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_examples += batch_size

    if total_examples == 0:
        raise ValueError("dataloader 不能为空")
    return total_loss / total_examples, total_correct / total_examples


def fit_tiny_classifier(features, labels, epochs=100, batch_size=4, learning_rate=0.1):
    """题5：训练 TinyClassifier 学会二维二分类。

    使用 make_dataloader、nn.CrossEntropyLoss、torch.optim.SGD、
    train_one_epoch 和 evaluate。

    为便于复现，请在创建模型前调用 torch.manual_seed(42)。
    返回 (训练后的 model, 在同一数据上的 Python float accuracy)。
    """
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyClassifier().to(device)
    dataloader = make_dataloader(
        features, labels, batch_size=batch_size, shuffle=True
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        train_one_epoch(model, dataloader, loss_fn, optimizer, device)

    _, accuracy = evaluate(model, dataloader, loss_fn, device)
    return model, accuracy


def save_model_weights(model, path):
    """题6a：只保存模型 state_dict 到 path。"""
    path = Path(path)
    torch.save(model.state_dict(), path)


def load_model_weights(model, path):
    """题6b：从 path 加载权重到已创建的同结构 model 并返回。

    要求能把 CPU 权重安全加载到当前这个练习模型。
    """
    path = Path(path)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model


# ========== 自动判分（不要修改） ==========

def _toy_data():
    features = torch.tensor([
        [-2.0, -1.0],
        [-1.0, -2.0],
        [-1.5, -0.5],
        [-0.5, -1.5],
        [1.0, 2.0],
        [2.0, 1.0],
        [1.5, 0.5],
        [0.5, 1.5],
    ])
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int64)
    return features, labels


def _check_dataset():
    features, labels = _toy_data()
    dataset = PointDataset(features, labels)
    feature, label = dataset[2]
    mismatch_rejected = False
    try:
        PointDataset(features, labels[:-1])
    except (ValueError, AssertionError):
        mismatch_rejected = True
    return (
        len(dataset) == 8
        and torch.equal(feature, features[2])
        and torch.equal(label, labels[2])
        and mismatch_rejected
    )


def _check_dataloader():
    features = torch.arange(14, dtype=torch.float32).reshape(7, 2)
    labels = torch.arange(7)
    loader = make_dataloader(features, labels, batch_size=3, shuffle=False)
    batches = list(loader)
    batch_sizes = [batch_features.shape[0] for batch_features, _ in batches]
    recovered_labels = torch.cat([batch_labels for _, batch_labels in batches])
    return (
        isinstance(loader, DataLoader)
        and batch_sizes == [3, 3, 1]
        and torch.equal(recovered_labels, labels)
    )


def _check_train_epoch():
    features, labels = _toy_data()
    loader = make_dataloader(features, labels, batch_size=3, shuffle=False)
    torch.manual_seed(3)
    model = TinyClassifier()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    average_loss = train_one_epoch(model, loader, loss_fn, optimizer, torch.device("cpu"))
    changed = any(
        not torch.allclose(old, new.detach())
        for old, new in zip(before, model.parameters())
    )
    return (
        model.training
        and isinstance(average_loss, float)
        and average_loss > 0
        and torch.isfinite(torch.tensor(average_loss))
        and changed
    )


def _check_evaluate():
    features = torch.tensor([
        [2.0, 0.0],
        [1.0, 3.0],
        [-1.0, 2.0],
        [-3.0, 0.0],
    ])
    labels = torch.tensor([0, 0, 1, 1])
    loader = make_dataloader(features, labels, batch_size=3, shuffle=False)
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
        model.bias.zero_()
    for parameter in model.parameters():
        parameter.grad = None
    loss, accuracy = evaluate(
        model, loader, nn.CrossEntropyLoss(), torch.device("cpu")
    )
    no_grad_created = all(parameter.grad is None for parameter in model.parameters())
    return (
        not model.training
        and isinstance(loss, float)
        and isinstance(accuracy, float)
        and abs(accuracy - 1.0) < 1e-8
        and no_grad_created
    )


def _check_fit():
    features, labels = _toy_data()
    model, accuracy = fit_tiny_classifier(features, labels)
    return isinstance(model, TinyClassifier) and accuracy >= 0.99


def _check_save_load():
    from tempfile import TemporaryDirectory

    torch.manual_seed(11)
    source = TinyClassifier()
    torch.manual_seed(23)
    target = TinyClassifier()
    features, _ = _toy_data()

    with TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / "weights.pth"
        save_model_weights(source, path)
        if not path.exists():
            return False
        loaded = load_model_weights(target, path)

    return (
        loaded is target
        and torch.allclose(source(features), target(features))
    )


def _check():
    tests = [
        ("题1 自定义Dataset", _check_dataset),
        ("题2 DataLoader分批", _check_dataloader),
        ("题3 训练一个epoch", _check_train_epoch),
        ("题4 验证loss与accuracy", _check_evaluate),
        ("题5 完整二分类训练", _check_fit),
        ("题6 保存并加载权重", _check_save_load),
    ]

    print("\nDay6 数据管道与训练/验证练习自动判分")
    print("=" * 56)
    passed = 0
    for name, test in tests:
        try:
            ok = bool(test())
        except Exception as exc:
            ok = False
            name += f"  (报错: {type(exc).__name__}: {exc})"
        print("  " + ("[OK]" if ok else "[--]") + "  " + name)
        passed += ok
    print("=" * 56)
    if passed == len(tests):
        print(f"通过 {passed}/{len(tests)}  全部完成！你已具备进入 MNIST 的完整地基。")
    else:
        print(f"通过 {passed}/{len(tests)}  先检查 batch 大小、模式切换和指标分母。")


if __name__ == "__main__":
    _check()
