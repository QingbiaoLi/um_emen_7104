"""Student assignment: hand-written recurrent vehicle trajectory prediction.

Complete all TODOs, then run: python student.py --epochs 40 --output outputs/student
See ASSIGNMENT_zh.md for equations, tensor shapes and the marking rubric.
No nn.RNN, nn.GRU, nn.LSTM or their Cell variants are used.
"""

import torch
from torch import nn

from trajectory_utils import run_experiment


def make_features(observed, dt):
    """observed: (B,O,2) meters -> features: (B,O,4), using history only."""
    # TODO [features]: Concatenate last-position-centered coordinates and finite-difference velocities; copy velocity at index 1 into index 0.
    raise NotImplementedError("TODO: features")


class ElmanRNN(nn.Module):
    def __init__(self, d_x, d_h, d_y):
        super().__init__()
        # TODO [elman_init]: Store d_h; create input/recurrent/output Linear layers, orthogonal recurrent weights and a zero-initialized output head.
        raise NotImplementedError("TODO: elman_init")

    def f(self, h, x):
        """h: (B,d_h), x: (B,d_x) -> next hidden state: (B,d_h)."""
        # TODO [elman_f]: Apply tanh to the sum of input and hidden affine projections.
        raise NotImplementedError("TODO: elman_f")

    def g(self, h):
        """Map final hidden state to all future residual coordinates."""
        # TODO [elman_g]: Map hidden states to d_y output coordinates.
        raise NotImplementedError("TODO: elman_g")

    def initial_recurrent_state(self, batch_size):
        # TODO [elman_h0]: Return (batch_size, d_h) zeros on the parameter device and dtype.
        raise NotImplementedError("TODO: elman_h0")


def scan(f, h, xs):
    """Scan time, retaining autograd: (B,O,d_x) -> (B,O,d_h)."""
    # TODO [scan]: Loop over time dimension 1, update h, and stack states without detaching gradients.
    raise NotImplementedError("TODO: scan")


class MyRNN(nn.Module):
    """A manually implemented gated RNN (GRU equations)."""

    def __init__(self, d_x, d_h, d_y):
        super().__init__()
        # TODO [custom_init]: Implement your gated or other improved recurrent parameters; use a zero-initialized output head.
        raise NotImplementedError("TODO: custom_init")

    def f(self, h, x):
        # TODO [custom_f]: Implement your recurrent update, for example reset/update gates and a candidate state.
        raise NotImplementedError("TODO: custom_f")

    def g(self, h):
        # TODO [custom_g]: Map hidden states to d_y output coordinates.
        raise NotImplementedError("TODO: custom_g")

    def initial_recurrent_state(self, batch_size):
        # TODO [custom_h0]: Return (batch_size, d_h) zeros on the parameter device and dtype.
        raise NotImplementedError("TODO: custom_h0")


def predict(rnn, xs, baseline, output_scale):
    """Normalized history + CV forecast -> future relative positions in meters.

    xs: (B,O,4); baseline: (B,H,2); output_scale: a fixed 10-meter scale.
    No future ground-truth values are available to this function.
    """
    # TODO [predict]: Use initial_recurrent_state, scan, and g on the final state; reshape to (B,H,2), scale, and add baseline.
    raise NotImplementedError("TODO: predict")


def train_one_epoch(rnn, loader, optimizer, device, output_scale):
    """One epoch; return sample-weighted normalized coordinate MSE."""
    # TODO [train]: Train one epoch: zero_grad, predict, normalized MSE, backward, gradient clipping, optimizer.step; return sample-weighted mean loss.
    raise NotImplementedError("TODO: train")


def displacement_errors(prediction, target):
    """Return per-window ADE, FDE: two tensors of shape (B,), in meters."""
    # TODO [metrics]: Compute Euclidean distances; return the time mean and final-step distance for each window.
    raise NotImplementedError("TODO: metrics")


if __name__ == "__main__":
    run_experiment(
        {"elman": ElmanRNN, "gated": MyRNN}, make_features,
        predict, train_one_epoch, displacement_errors,
    )
