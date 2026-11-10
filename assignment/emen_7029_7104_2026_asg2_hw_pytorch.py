# -*- coding: utf-8 -*-
"""EMEN_7029_7104_2024_asg2_hw_pytorch.py

PyTorch conversion of the original JAX assignment.

Here, we will implement our first neural network! You will need to accomplish
a number of tasks:

1. Implement a linear layer in `linear` (15 points)
2. Implement a perceptron (single layer neural network) in `f` (15 points)
3. Implement the `loss_single` function, which is the loss function for a
   single x_[i], y_[i] datapoint. (15 points)

Then, you will need to use what we learned about optimization and
backpropagation. You must

4. Implement the `single_grad_loss`, which computes the gradient of the
   `loss_single` function. Written mathematically, you must compute
   ∇_theta L(x, y, (W, b)). You may NOT use PyTorch autograd inside
   `single_grad_loss`. You need to figure out the gradient by hand. (25 points)
5. Ensure your functions are correct, such that the error computed between
   your `single_grad_loss` and PyTorch autograd is less than 0.001. (30 points)

This assignment will be graded out of 100 points.
"""

import torch
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Set up some variables for our experiments
# ---------------------------------------------------------------------------

n = 32       # number of x_[i], y_[i] datapoints
d_x = 2      # size of x
d_y = 1      # size of y


# ---------------------------------------------------------------------------
# Define our neural network
# ---------------------------------------------------------------------------

def sigmoid(z: torch.Tensor) -> torch.Tensor:
    """Activation function, a differentiable approximation of the step function."""
    return torch.sigmoid(z)


def linear(x: torch.Tensor, theta) -> torch.Tensor:
    """Linear layer."""
    # TODO: Implement a linear layer.
    #
    # theta = (W, b)
    # W shape: (d_x, d_y)
    # b shape: (d_y,)
    #
    # QUESTION:
    # What equation should be implemented here?
    #
    # Hint:
    # z = x @ W + b

    W, b = theta

    # Placeholder with the same output shape as b.
    return torch.zeros_like(b)


def f(x: torch.Tensor, theta) -> torch.Tensor:
    """A single-layer perceptron (neural network)."""
    # TODO: Create a single-layer neural network (perceptron).
    #
    # QUESTION:
    # How do `linear` and `sigmoid` compose?
    #
    # Hint:
    # z = linear(x, theta)
    # y_pred = sigmoid(z)

    _, b = theta

    # Placeholder with output shape (d_y,).
    return torch.zeros_like(b)


def debug_shape(x):
    """Helpful function for printing tensor shapes."""
    if isinstance(x, (tuple, list)):
        print(tuple(item.shape for item in x))
    else:
        print(x.shape)


# ---------------------------------------------------------------------------
# Define our losses
# ---------------------------------------------------------------------------

def loss_single(
    x: torch.Tensor,
    y: torch.Tensor,
    theta
) -> torch.Tensor:
    """Loss function for a single datapoint x_i, y_i."""
    # TODO: Define the squared-error loss for a single x_i, y_i.
    #
    # QUESTION:
    # If y_pred = f(x, theta), what is the squared-error loss?
    #
    # Common form:
    # L = sum((y - y_pred) ** 2)

    return torch.tensor(1.0, dtype=x.dtype, device=x.device)


def loss(
    x: torch.Tensor,
    y: torch.Tensor,
    theta
) -> torch.Tensor:
    """Loss function over the complete dataset."""
    # JAX used:
    #   jax.vmap(loss_single, in_axes=(0, 0, None))
    #
    # In PyTorch, for this small teaching example, we can simply evaluate the
    # single-sample loss for every datapoint and stack the results.

    losses = torch.stack([
        loss_single(x_i, y_i, theta)
        for x_i, y_i in zip(x, y)
    ])

    return torch.sum(losses)


# ---------------------------------------------------------------------------
# Manual gradient
# ---------------------------------------------------------------------------

def single_grad_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    theta
):
    """Compute dL/dtheta for a single datapoint x_i, y_i.

    IMPORTANT:
    Do NOT use torch.autograd.grad() or .backward() inside this function.
    Compute the gradient analytically using the chain rule.
    """

    W, b = theta

    # You will need these values to compute the gradient.
    z = linear(x, theta)
    y_pred = sigmoid(z)

    # TODO: Compute gradients manually.
    #
    # For squared error:
    #
    #   L = (y - y_pred)^2
    #   y_pred = sigmoid(z)
    #   z = x @ W + b
    #
    # Work out:
    #   dL/dy_pred
    #   dy_pred/dz
    #   dz/dW
    #   dz/db
    #
    # Then apply the chain rule.
    #
    # Expected return:
    #   grad_W: same shape as W = (d_x, d_y)
    #   grad_b: same shape as b = (d_y,)

    grad_W = torch.ones_like(W)
    grad_b = torch.ones_like(b)

    return grad_W, grad_b


def grad_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    theta
):
    """Compute the manually derived gradient over all datapoints."""

    sample_grads = [
        single_grad_loss(x_i, y_i, theta)
        for x_i, y_i in zip(x, y)
    ]

    grad_W = torch.stack([g[0] for g in sample_grads]).sum(dim=0)
    grad_b = torch.stack([g[1] for g in sample_grads]).sum(dim=0)

    return grad_W, grad_b


# ---------------------------------------------------------------------------
# Generate our dataset X and Y
# ---------------------------------------------------------------------------

def generate_regression_data(
    seed: int,
    n_samples: int,
    noise: float = 0.1
):
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Random points in [-1, 1].
    X = torch.rand(
        (n_samples, d_x),
        generator=generator,
        dtype=torch.float32
    ) * 2.0 - 1.0

    y = torch.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)

    y = y + noise * torch.randn(
        (n_samples,),
        generator=generator,
        dtype=torch.float32
    )

    # Normalize to [0, 1].
    y = (y - y.min()) / (y.max() - y.min())

    return X, y


x, y = generate_regression_data(seed=0, n_samples=n)


# ---------------------------------------------------------------------------
# Randomly initialize the parameters of our neural network
# ---------------------------------------------------------------------------

weight_generator = torch.Generator()
weight_generator.manual_seed(1)

bias_generator = torch.Generator()
bias_generator.manual_seed(2)

W = torch.rand(
    (d_x, d_y),
    generator=weight_generator,
    dtype=torch.float32
) - 0.5

b = torch.rand(
    (d_y,),
    generator=bias_generator,
    dtype=torch.float32
) - 0.5

theta = (W, b)


# ---------------------------------------------------------------------------
# Verify that the functions run
# ---------------------------------------------------------------------------

f(x[0], theta)

loss_single(x[0], y[0], theta)
loss(x, y, theta)


# ---------------------------------------------------------------------------
# Verify the manual gradient for a SINGLE sample
# ---------------------------------------------------------------------------

# PyTorch only computes gradients for tensors with requires_grad=True.
W_auto = W.detach().clone().requires_grad_(True)
b_auto = b.detach().clone().requires_grad_(True)
theta_auto = (W_auto, b_auto)

your_grad_single = single_grad_loss(x[0], y[0], theta)

single_loss_for_autograd = loss_single(x[0], y[0], theta_auto)

auto_grad_single = torch.autograd.grad(
    single_loss_for_autograd,
    theta_auto,
    create_graph=False
)

print("Automatic gradient:")
print(auto_grad_single)

print("Your gradient:")
print(your_grad_single)

grad_difference_single = tuple(
    auto - manual
    for auto, manual in zip(auto_grad_single, your_grad_single)
)

print("Difference in gradient:")
print(grad_difference_single)

grad_error_single = torch.linalg.vector_norm(
    torch.cat([
        diff.reshape(-1)
        for diff in grad_difference_single
    ])
)

print("Gradient error:", grad_error_single.item())

if grad_error_single < 1e-3:
    print("Gradient correct!")
else:
    print("Gradient incorrect!")


# ---------------------------------------------------------------------------
# Verify the manual gradient over the ENTIRE dataset
# ---------------------------------------------------------------------------

W_auto = W.detach().clone().requires_grad_(True)
b_auto = b.detach().clone().requires_grad_(True)
theta_auto = (W_auto, b_auto)

your_grad = grad_loss(x, y, theta)

dataset_loss_for_autograd = loss(x, y, theta_auto)

auto_grad = torch.autograd.grad(
    dataset_loss_for_autograd,
    theta_auto,
    create_graph=False
)

print("Automatic gradient:")
print(auto_grad)

print("Your gradient:")
print(your_grad)

grad_difference = tuple(
    auto - manual
    for auto, manual in zip(auto_grad, your_grad)
)

print("Difference in gradient:")
print(grad_difference)

grad_error = torch.linalg.vector_norm(
    torch.cat([
        diff.reshape(-1)
        for diff in grad_difference
    ])
)

print("Gradient error:", grad_error.item())

if grad_error < 1e-3:
    print("Gradient correct!")
else:
    print("Gradient incorrect!")
