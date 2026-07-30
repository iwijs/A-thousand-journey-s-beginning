import math
import random
import unittest

from minigrad import MLP, Value


class ValueTests(unittest.TestCase):
    def test_chain_rule(self):
        x = Value(2.0)
        y = Value(-3.0)
        output = (x * y + x**2).tanh()
        output.backward()

        inner = x.data * y.data + x.data**2
        local = 1.0 - math.tanh(inner) ** 2
        self.assertAlmostEqual(x.grad, (y.data + 2.0 * x.data) * local)
        self.assertAlmostEqual(y.grad, x.data * local)

    def test_gradient_accumulates_along_multiple_paths(self):
        x = Value(3.0)
        output = x * x + x
        output.backward()
        self.assertAlmostEqual(x.grad, 7.0)

    def test_mlp_parameters_receive_gradients(self):
        random.seed(7)
        model = MLP(2, [3, 1])
        prediction = model([1.0, -2.0])
        self.assertIsInstance(prediction, Value)
        loss = (prediction - 1.0) ** 2
        loss.backward()
        self.assertTrue(any(abs(parameter.grad) > 0 for parameter in model.parameters()))

    def test_sigmoid_value_and_gradient(self):
        x = Value(0.0)
        output = x.sigmoid()
        output.backward()

        self.assertAlmostEqual(output.data, 0.5)
        self.assertAlmostEqual(x.grad, 0.25)

    def test_gradient_matches_finite_difference(self):
        def forward(x_value, y_value):
            x = Value(x_value)
            y = Value(y_value)
            return (x * y + x**2).tanh().data

        x_value = 2.0
        y_value = -3.0
        epsilon = 1e-6

        x = Value(x_value)
        y = Value(y_value)
        output = (x * y + x**2).tanh()
        output.backward()

        numerical_x_grad = (
            forward(x_value + epsilon, y_value)
            - forward(x_value - epsilon, y_value)
        ) / (2.0 * epsilon)

        numerical_y_grad = (
            forward(x_value, y_value + epsilon)
            - forward(x_value, y_value - epsilon)
        ) / (2.0 * epsilon)

        self.assertAlmostEqual(x.grad, numerical_x_grad, places=6)
        self.assertAlmostEqual(y.grad, numerical_y_grad, places=6)


if __name__ == "__main__":
    unittest.main()
