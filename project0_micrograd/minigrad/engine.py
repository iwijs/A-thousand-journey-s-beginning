"""Scalar automatic differentiation.

This module intentionally favors readability over speed.  Each ``Value`` stores
one scalar and one local backward function. ``backward`` topologically sorts the
graph, then applies the chain rule in reverse order.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable


class Value:
    """A scalar value and its gradient in a dynamically built computation graph."""

    def __init__(
        self,
        data: float,
        children: Iterable["Value"] = (),
        op: str = "",
        label: str = "",
    ):
        self.data = float(data)
        self.grad = 0.0
        self.label = label
        self._op = op
        self._prev = set(children)
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"

    @staticmethod
    def _coerce(other: float | "Value") -> "Value":
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other: float | "Value") -> "Value":
        other = self._coerce(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: float | "Value") -> "Value":
        return self + other

    def __mul__(self, other: float | "Value") -> "Value":
        other = self._coerce(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: float | "Value") -> "Value":
        return self * other

    def __pow__(self, exponent: int | float) -> "Value":
        if not isinstance(exponent, (int, float)):
            raise TypeError("Value only supports a numeric exponent")
        out = Value(self.data**exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> "Value":
        return self * -1.0

    def __sub__(self, other: float | "Value") -> "Value":
        return self + (-self._coerce(other))

    def __rsub__(self, other: float | "Value") -> "Value":
        return self._coerce(other) - self

    def __truediv__(self, other: float | "Value") -> "Value":
        return self * self._coerce(other) ** -1

    def __rtruediv__(self, other: float | "Value") -> "Value":
        return self._coerce(other) / self

    def tanh(self) -> "Value":
        value = math.tanh(self.data)
        out = Value(value, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - value**2) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> "Value":
        value = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(value, (self,), "sigmoid")

        def _backward() -> None:
            self.grad += value * (1.0 - value) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        value = max(0.0, self.data)
        out = Value(value, (self,), "ReLU")

        def _backward() -> None:
            self.grad += (1.0 if self.data > 0.0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Value":
        value = math.exp(self.data)
        out = Value(value, (self,), "exp")

        def _backward() -> None:
            self.grad += value * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Compute gradients of this scalar with respect to every ancestor."""

        topological_order: list[Value] = []
        visited: set[Value] = set()

        def build(node: Value) -> None:
            if node in visited:
                return
            visited.add(node)
            for child in node._prev:
                build(child)
            topological_order.append(node)

        build(self)

        self.grad = 1.0

        for node in reversed(topological_order):
            node._backward()
