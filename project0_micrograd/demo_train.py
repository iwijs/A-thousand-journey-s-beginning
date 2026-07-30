"""Train a tiny scalar MLP on four two-dimensional samples."""

import argparse
import random

from minigrad import MLP, Value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help=(
            "use a constant learning rate; when omitted, use the default "
            "0.08-to-0.04 decay schedule"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(42)
    features = [
        [2.0, 3.0],
        [3.0, -1.0],
        [-2.0, 1.0],
        [-3.0, -2.0],
    ]
    targets = [1.0, 1.0, -1.0, -1.0]
    model = MLP(2, [4, 4, 1])

    for step in range(200):
        predictions = [model(sample) for sample in features]
        assert all(isinstance(prediction, Value) for prediction in predictions)
        loss = sum(
            (prediction - target) ** 2
            for prediction, target in zip(predictions, targets)
        ) / len(targets)

        model.zero_grad()
        loss.backward()

        learning_rate = (
            args.learning_rate
            if args.learning_rate is not None
            else 0.08 * (1.0 - 0.5 * step / 200)
        )
        for parameter in model.parameters():
            parameter.data -= learning_rate * parameter.grad

        if step % 25 == 0 or step == 199:
            accuracy = sum(
                (prediction.data > 0) == (target > 0)
                for prediction, target in zip(predictions, targets)
            ) / len(targets)
            print(
                f"step={step:03d} loss={loss.data:.6f} "
                f"accuracy={accuracy:.0%}"
            )


if __name__ == "__main__":
    main()
