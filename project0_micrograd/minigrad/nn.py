"""Tiny neural-network layers built from scalar ``Value`` objects."""

from __future__ import annotations

import random
from typing import Iterable, Sequence

from .engine import Value


class Module:
    def parameters(self) -> list[Value]:
        return []

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.grad = 0.0


class Neuron(Module):
    def __init__(self, num_inputs: int, nonlinearity: bool = True):
        self.weights = [Value(random.uniform(-1.0, 1.0)) for _ in range(num_inputs)]
        self.bias = Value(0.0)
        self.nonlinearity = nonlinearity

    def __call__(self, inputs: Sequence[float | Value]) -> Value:
        if len(inputs) != len(self.weights):
            raise ValueError(
                f"expected {len(self.weights)} inputs, received {len(inputs)}"
            )
        activation = sum(
            (weight * value for weight, value in zip(self.weights, inputs)),
            self.bias,
        )
        return activation.tanh() if self.nonlinearity else activation

    def parameters(self) -> list[Value]:
        return [*self.weights, self.bias]


class Layer(Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        nonlinearity: bool = True,
    ):
        self.neurons = [
            Neuron(num_inputs, nonlinearity=nonlinearity)
            for _ in range(num_outputs)
        ]

    def __call__(self, inputs: Sequence[float | Value]) -> Value | list[Value]:
        outputs = [neuron(inputs) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self) -> list[Value]:
        return [
            parameter
            for neuron in self.neurons
            for parameter in neuron.parameters()
        ]


class MLP(Module):
    def __init__(self, num_inputs: int, layer_sizes: Iterable[int]):
        layer_sizes = list(layer_sizes)
        sizes = [num_inputs, *layer_sizes]
        self.layers = [
            Layer(
                sizes[index],
                sizes[index + 1],
                nonlinearity=index != len(layer_sizes) - 1,
            )
            for index in range(len(layer_sizes))
        ]

    def __call__(self, inputs: Sequence[float | Value]) -> Value | list[Value]:
        output: Value | list[Value] | Sequence[float | Value] = inputs
        for layer in self.layers:
            if isinstance(output, Value):
                output = [output]
            output = layer(output)
        return output

    def parameters(self) -> list[Value]:
        return [
            parameter
            for layer in self.layers
            for parameter in layer.parameters()
        ]
