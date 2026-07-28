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


if __name__ == "__main__":
    unittest.main()
