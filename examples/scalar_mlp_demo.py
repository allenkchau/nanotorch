"""
Small scalar MLP demo.

Toy dataset:
- 3 input features
- binary targets in {-1, 1}
"""

from nanotorch.nn.mlp import MLP
from nanotorch.nn.activation import relu

import numpy as np

# consistency across runs
np.random.seed(42)

# toy classification dataset 
# each sample has 3 scalar features and one scalar target
X = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
y = [1.0, -1.0, -1.0, 1.0]

# zipped dataset
train_data = list(zip(X, y))

# mlp model
model = MLP(layer_sizes=[3, 4, 4, 1], activation=relu)

# define the loss function
def MSELoss(pred, target):
    error = pred - target
    squared_error = error * error
    return squared_error


def compute_dataset_loss():
    loss = 0
    for x_i, y_i in train_data:
        y_pred = model(x_i)[0]
        loss += MSELoss(y_pred, y_i)
    return loss / len(train_data)


def train_step(lr):
    loss = compute_dataset_loss()

    for p in model.parameters():
        p.grad = 0

    loss.backward()

    for p in model.parameters():
        p.data -= lr * p.grad

    return loss


def main():
    # hyperparameters
    lr = 0.01
    epochs = 200

    # 1) initial loss
    initial_loss = compute_dataset_loss().data
    print(f"Initial loss: {initial_loss:.6f}")

    # 2) one update step + before/after parameter check
    tracked_param = model.parameters()[0]
    before = tracked_param.data
    step0_loss = train_step(lr).data
    after = tracked_param.data
    loss_after_one = compute_dataset_loss().data

    print(f"Loss used for first update: {step0_loss:.6f}")
    print(f"Loss after one update: {loss_after_one:.6f}")
    print(
        "Tracked param[0] before/after one step: "
        f"{before:.6f} -> {after:.6f} (delta={after - before:.6e})"
    )

    # 3) many updates
    for epoch in range(1, epochs):
        loss = train_step(lr)
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:03d} | loss: {loss.data:.6f}")

    loss_after_many = compute_dataset_loss().data
    print(f"Loss after many updates: {loss_after_many:.6f}")



if __name__ == "__main__":
    main()


