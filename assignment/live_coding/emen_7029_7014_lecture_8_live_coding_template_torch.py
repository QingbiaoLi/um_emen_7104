# -*- coding: utf-8 -*-
"""EMEN 7029 / 7014, Lecture 8: recurrent counting models in PyTorch.

Converted from the JAX/Equinox live-coding template. Both counting examples
and the shape-typing demonstration are retained.

Install dependencies once in a terminal:
    python -m pip install torch jaxtyping beartype

In a Colab/Jupyter code cell, use:
    %pip install torch jaxtyping beartype

The # %% markers separate the original teaching sections for cell-based use.
"""

# %% Imports and tensor shapes
from typing import Tuple

import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn


# jaxtyping also supports PyTorch tensors; no JAX dependency is needed.
# The dimension names describe shapes and bind within each checked call.
Input = Float[Tensor, "d_x"]
RecurrentState = Float[Tensor, "d_h"]
Output = Float[Tensor, "d_y"]

Inputs = Float[Tensor, "T d_x"]
RecurrentStates = Float[Tensor, "T d_h"]


# %% Example 1: separate recurrence and recall networks
# Today, we will teach a recurrent function to count to ten.
T = 10
d_x = 1
d_y = 1
d_h = 64

# Only a single training example, n = 1.
xs = torch.ones((T, d_x), dtype=torch.float32)
ys = torch.arange(1, T + 1, dtype=torch.float32).reshape(T, d_y)


class Recurrence(nn.Module):
    def __init__(self, d_x: int, d_h: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_x, d_h),
            # Tanh bounds the input contribution; accumulated states can grow.
            nn.Tanh(),
        )

    @jaxtyped(typechecker=beartype)
    def forward(self, h: RecurrentState, x: Inputs) -> RecurrentStates:
        # PyTorch records the operations in this loop for backpropagation
        # through time. Do not detach h during the recurrence.
        hs = []
        for x_t in x:
            h = F.leaky_relu(h, negative_slope=0.01) + self.network(x_t)
            hs.append(h)

        # This replaces the sequence output of jax.lax.scan.
        return torch.stack(hs, dim=0)  # Shape: (T, d_h).


class Recall(nn.Module):
    def __init__(self, d_h: int, d_y: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_h, d_h),
            # Match the tanh approximation used by the original JAX GELU.
            nn.GELU(approximate="tanh"),
            nn.Linear(d_h, d_y),
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        h: Float[Tensor, "*batch d_h"],
        x: Float[Tensor, "*batch d_x"],
    ) -> Float[Tensor, "*batch d_y"]:
        # Keep x in the signature to match g(h, x) in the original lesson.
        # This particular recall network depends only on h.
        # nn.Linear handles either (d_h,) or (T, d_h) directly.
        return self.network(h)


# Test the model. Seeds make PyTorch runs repeatable, but do not reproduce
# JAX's random initial weights exactly.
torch.manual_seed(0)
scan_f = Recurrence(d_x, d_h)
torch.manual_seed(1)
g = Recall(d_h, d_y)
h0 = torch.zeros(d_h, dtype=torch.float32)

with torch.no_grad():
    hs = scan_f(h0, xs)
    ys_pred = g(hs, xs)  # Shape: (T, d_y); no filter_vmap is needed.

models = (scan_f, g)

# Create the optimizer and tell it which parameters to update.
# Explicitly match Optax AdamW's default weight_decay = 1e-4.
optim = torch.optim.AdamW(
    list(scan_f.parameters()) + list(g.parameters()),
    lr=0.001,
    weight_decay=1e-4,
)


def L_separate(models, h0: Tensor, xs: Tensor, ys: Tensor) -> Tensor:
    scan_f, g = models
    hs = scan_f(h0, xs)
    ys_pred = g(hs, xs)
    # Retain the original SUM of squared errors, rather than mean error.
    return torch.sum((ys_pred - ys) ** 2)


# Now, train! The optimizer retains its momentum/variance state internally.
print("Example 1: count from 1 to 10")
for epoch in range(100):
    # PyTorch accumulates gradients unless we explicitly clear them.
    optim.zero_grad()
    loss = L_separate(models, h0, xs, ys)
    loss.backward()
    optim.step()

    if epoch == 0 or (epoch + 1) % 10 == 0:
        # Evaluate the UPDATED parameters without building a gradient graph.
        with torch.no_grad():
            hs = scan_f(h0, xs)
            ys_pred = g(hs, xs)
            eval_loss = torch.sum((ys_pred - ys) ** 2).item()
            print(f"Epoch: {epoch + 1}, loss: {eval_loss:.6f}")
            print(torch.round(ys_pred))
        # input("Press enter to continue")


# %% Shape-typing demonstration using PyTorch
# Fix the dimension in this demonstration so later d_x changes do not affect it.
DemoInput = Float[Tensor, "10"]


@jaxtyped(typechecker=beartype)
def f(x: DemoInput) -> DemoInput:
    return x


f(torch.zeros(10, dtype=torch.float32))
# Try f(torch.zeros(9)) to see a runtime shape-checking error.


# %% Example 2: recurrence and recall in one module
# This example extends the task to counting from 1 to 20.
T = 20
d_x = 1
d_y = 1
d_h = 64

# Only a single training example, n = 1.
xs = torch.ones((T, d_x), dtype=torch.float32)
ys = torch.arange(1, T + 1, dtype=torch.float32).reshape(T, d_y)


class RecurrentModel(nn.Module):
    def __init__(self, d_x: int, d_h: int, d_y: int):
        super().__init__()
        self.f_network = nn.Sequential(
            nn.Linear(d_x, d_h),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.g_network = nn.Sequential(
            nn.Linear(d_h, d_y),
        )

    @jaxtyped(typechecker=beartype)
    def f(
        self, h: RecurrentState, x: Input
    ) -> Tuple[RecurrentState, RecurrentState]:
        # You need to implement a different f in the homework.
        # Here, we implement the same simple recurrence as in the template.
        result = F.leaky_relu(h, negative_slope=0.01) + self.f_network(x)

        # Retain the original scan convention for teaching:
        # one value is the next carry; the other is this timestep's output.
        return result, result

    @jaxtyped(typechecker=beartype)
    def g(
        self,
        h: Float[Tensor, "*batch d_h"],
        x: Float[Tensor, "*batch d_x"],
    ) -> Float[Tensor, "*batch d_y"]:
        return self.g_network(h)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, h: RecurrentState, x: Inputs
    ) -> Tuple[RecurrentState, RecurrentStates]:
        # Replaces lax.scan(model.f, h0, xs), including the unfinished
        # partition_scan helper in the original template.
        hs = []
        for x_t in x:
            h, h_t = self.f(h, x_t)
            hs.append(h_t)

        # h is the final carry (d_h,); hs contains all states (T, d_h).
        return h, torch.stack(hs, dim=0)


torch.manual_seed(0)
model = RecurrentModel(d_x, d_h, d_y)

# Test one recurrent step and its output.
h0 = torch.zeros(d_h, dtype=torch.float32)
x1 = torch.zeros(d_x, dtype=torch.float32)
with torch.no_grad():
    next_h, h1 = model.f(h0, x1)
    y1 = model.g(h1, x1)

# model.parameters() includes both f_network and g_network.
optim = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)


def L(
    model: RecurrentModel, h0: Tensor, xs: Tensor, ys: Tensor
) -> Tuple[Tensor, Tensor]:
    # One final state and one state per timestep, just as in the JAX scan.
    hT, hs = model(h0, xs)
    ys_pred = model.g(hs, xs)
    loss = torch.sum((ys_pred - ys) ** 2)
    return loss, ys_pred


# Now, train! loss.backward() replaces filter_grad; step() applies updates.
# Eager-mode PyTorch needs no JIT wrapper for this lesson.
print("\nExample 2: count from 1 to 20")
for epoch in range(100):
    optim.zero_grad()
    loss, ys_pred = L(model, h0, xs, ys)
    loss.backward()
    optim.step()

    if epoch == 0 or (epoch + 1) % 20 == 0:
        with torch.no_grad():
            eval_loss, ys_pred = L(model, h0, xs, ys)
            print(f"Epoch: {epoch + 1}, loss: {eval_loss.item():.6f}")
            print(ys_pred)
        # input("Press enter to continue")
