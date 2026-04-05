"""Small scalar MLP demo with training diagnostics."""

import numpy as np

from nanotorch.nn.activation import relu
from nanotorch.nn.mlp import MLP
from nanotorch.optim.sgd import SGD
from nanotorch.value import Value

SEED = 42
LR = 0.01
EPOCHS = 200
LOG_EVERY = 20


def make_toy_dataset():
    """Return tiny 3-feature binary classification dataset."""
    features = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    targets = [1.0, -1.0, -1.0, 1.0]
    return list(zip(features, targets))


def mse_loss(pred: Value, target: float) -> Value:
    error = pred - target
    return error * error


def compute_dataset_loss(model: MLP, train_data):
    loss = 0
    for features, target in train_data:
        pred = model(features)[0]
        loss += mse_loss(pred, target)
    return loss / len(train_data)


def train_step(model: MLP, optimizer: SGD, train_data):
    loss = compute_dataset_loss(model, train_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def main():
    np.random.seed(SEED)
    train_data = make_toy_dataset()
    model = MLP(layer_sizes=[3, 4, 4, 1], activation=relu)
    optimizer = SGD(model.parameters(), lr=LR)

    initial_loss = compute_dataset_loss(model, train_data).data
    print(f"Initial loss: {initial_loss:.6f}")

    tracked_param = model.parameters()[0]
    before = tracked_param.data
    first_step_loss = train_step(model, optimizer, train_data).data
    after = tracked_param.data
    loss_after_one = compute_dataset_loss(model, train_data).data

    print(f"Loss used for first update: {first_step_loss:.6f}")
    print(f"Loss after one update: {loss_after_one:.6f}")
    print(
        "Tracked param[0] before/after one step: "
        f"{before:.6f} -> {after:.6f} (delta={after - before:.6e})"
    )

    for epoch in range(1, EPOCHS):
        loss = train_step(model, optimizer, train_data)
        if epoch % LOG_EVERY == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch:03d} | loss: {loss.data:.6f}")

    loss_after_many = compute_dataset_loss(model, train_data).data
    print(f"Loss after many updates: {loss_after_many:.6f}")


if __name__ == "__main__":
    main()


