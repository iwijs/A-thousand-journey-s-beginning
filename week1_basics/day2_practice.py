# -*- coding: utf-8 -*-
"""
Day2 练习：Python 特有语法 + NumPy 广播
======================================
把每个函数里的 TODO 补上，然后运行：
    python day2_practice.py
脚本会自动判分，告诉你每题通过还是没过。全绿就算过关。

有 C 基础，重点体会这些 Python 特有的点：切片、字典、推导式、类、以及 NumPy 广播。
卡住就把题号和你的写法发给 Claude。
"""

import numpy as np


# ========== Part A - Python 特有语法 ==========

def reverse_and_slice(lst):
    """题1：用【切片】把列表反转后返回（不许用 lst.reverse()）。
    例：[1,2,3] -> [3,2,1]。 提示：a[::-1]"""
    # 解答：用切片实现
    return lst[::-1]


def word_count(text):
    """题2：统计每个单词出现次数，返回字典 {单词: 次数}。
    单词按空格切分。例：'a b a' -> {'a':2, 'b':1}。 提示：str.split() + dict"""
    # 解答
    counts = {}
    for word in text.split():
        counts[word] = counts.get(word, 0) + 1
    return counts


def squares(n):
    """题3：用【列表推导式】返回 [0^2, 1^2, ..., (n-1)^2]。
    例：n=4 -> [0,1,4,9]。 提示：[x*x for x in range(n)]"""
    # 解答
    return [x * x for x in range(n)]


class Neuron:
    """题4：实现一个最简单的"神经元"类（为以后的 nn.Module 铺垫）。
    - __init__(self, w, b)：把 w、b 存成实例属性 self.w、self.b
    - forward(self, x)：返回 w * x + b
    例：Neuron(2, 1).forward(3) == 7"""
    def __init__(self, w, b):
        # 解答：存 self.w, self.b
        self.w = w
        self.b = b

    def forward(self, x):
        # 解答：返回 w*x+b
        return self.w * x + self.b


# ========== Part B - NumPy 广播与索引 ==========

def normalize(x):
    """题5：标准化 —— 返回 (x - 均值) / 标准差。x 是 1 维 np.ndarray。
    提示：x.mean()、x.std()"""
    # 解答
    return (x - x.mean()) / x.std()


def add_row_vector(mat, vec):
    """题6：【广播】把向量 vec 加到矩阵 mat 的每一行上。
    mat 形状 (m, n)，vec 形状 (n,)，返回 (m, n)。
    例：[[1,2],[3,4]] + [10,20] -> [[11,22],[13,24]]。 提示：直接 mat + vec"""
    # 解答
    return mat + vec


def threshold(x, t):
    """题7：【布尔索引】返回 x 中所有 > t 的元素（1 维数组）。
    例：x=[1,5,2,8], t=3 -> [5,8]。 提示：x[x > t]"""
    # 解答
    return x[x > t]


# ========== 自动判分（不用改下面） ==========
def _check():
    tests = [
        ("题1 切片反转", lambda: reverse_and_slice([1, 2, 3]) == [3, 2, 1]),
        ("题2 词频统计", lambda: word_count("a b a") == {"a": 2, "b": 1}),
        ("题3 列表推导", lambda: squares(4) == [0, 1, 4, 9]),
        ("题4 Neuron类", lambda: Neuron(2, 1).forward(3) == 7),
        ("题5 标准化",   lambda: np.allclose(normalize(np.array([1.0, 2, 3])),
                                          np.array([-1.224744871, 0, 1.224744871]))),
        ("题6 广播加行", lambda: np.array_equal(add_row_vector(np.array([[1, 2], [3, 4]]),
                                              np.array([10, 20])),
                                              np.array([[11, 22], [13, 24]]))),
        ("题7 布尔索引", lambda: np.array_equal(threshold(np.array([1, 5, 2, 8]), 3),
                                            np.array([5, 8]))),
    ]
    print("\nDay2 练习自动判分")
    print("=" * 40)
    passed = 0
    for name, t in tests:
        try:
            ok = bool(t())
        except Exception as e:
            ok = False
            name += "  (报错: " + type(e).__name__ + ")"
        print("  " + ("[OK]" if ok else "[--]") + "  " + name)
        passed += ok
    print("=" * 40)
    tail = "  全部完成！commit 上去吧" if passed == len(tests) else "  —— 把没过的题号发给 Claude"
    print("通过 " + str(passed) + "/" + str(len(tests)) + tail)


if __name__ == "__main__":
    _check()
