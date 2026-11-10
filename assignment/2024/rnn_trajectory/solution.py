"""Instructor solution: hand-written recurrent vehicle trajectory prediction.

Run: python solution.py --epochs 40
The student version has exactly the same interfaces and experiment pipeline.
No nn.RNN, nn.GRU, nn.LSTM or their Cell variants are used.
"""

import torch
from torch import nn

from trajectory_utils import run_experiment


def make_features(observed, dt):
    """observed: (B,O,2) meters -> features: (B,O,4), using history only."""
    # BEGIN ANSWER features
    relative = observed - observed[:, -1:, :]
    velocity = torch.zeros_like(observed)
    velocity[:, 1:] = (observed[:, 1:] - observed[:, :-1]) / dt
    velocity[:, 0] = velocity[:, 1]
    return torch.cat((relative, velocity), dim=-1)
    # END ANSWER features


class ElmanRNN(nn.Module):
    def __init__(self, d_x, d_h, d_y):
        super().__init__()
        # BEGIN ANSWER elman_init
        self.d_h = d_h
        self.input = nn.Linear(d_x, d_h)
        self.recurrent = nn.Linear(d_h, d_h, bias=False)
        self.output = nn.Linear(d_h, d_y)
        nn.init.orthogonal_(self.recurrent.weight)
        # Start from the same constant-velocity baseline for both models.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        # END ANSWER elman_init

    def f(self, h, x):
        """h: (B,d_h), x: (B,d_x) -> next hidden state: (B,d_h)."""
        # BEGIN ANSWER elman_f
        return torch.tanh(self.input(x) + self.recurrent(h))
        # END ANSWER elman_f

    def g(self, h):
        """Map final hidden state to all future residual coordinates."""
        # BEGIN ANSWER elman_g
        return self.output(h)
        # END ANSWER elman_g

    def initial_recurrent_state(self, batch_size):
        # BEGIN ANSWER elman_h0
        return self.input.weight.new_zeros((batch_size, self.d_h))
        # END ANSWER elman_h0


def scan(f, h, xs):
    """Scan time, retaining autograd: (B,O,d_x) -> (B,O,d_h)."""
    # BEGIN ANSWER scan
    states = []
    for x in xs.unbind(dim=1):
        h = f(h, x)
        states.append(h)
    return torch.stack(states, dim=1)
    # END ANSWER scan


class MyRNN(nn.Module):
    """A manually implemented gated RNN (GRU equations)."""

    def __init__(self, d_x, d_h, d_y):
        super().__init__()
        # BEGIN ANSWER custom_init
        self.d_h = d_h
        self.input = nn.Linear(d_x, 3 * d_h)
        self.recurrent = nn.Linear(d_h, 3 * d_h, bias=False)
        self.output = nn.Linear(d_h, d_y)
        for weight in self.recurrent.weight.chunk(3, dim=0):
            nn.init.orthogonal_(weight)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        # END ANSWER custom_init

    def f(self, h, x):
        # BEGIN ANSWER custom_f
        x_r, x_z, x_n = self.input(x).chunk(3, dim=-1)
        h_r, h_z, h_n = self.recurrent(h).chunk(3, dim=-1)
        reset = torch.sigmoid(x_r + h_r)
        update = torch.sigmoid(x_z + h_z)
        candidate = torch.tanh(x_n + reset * h_n)
        return update * h + (1.0 - update) * candidate
        # END ANSWER custom_f

    def g(self, h):
        # BEGIN ANSWER custom_g
        return self.output(h)
        # END ANSWER custom_g

    def initial_recurrent_state(self, batch_size):
        # BEGIN ANSWER custom_h0
        return self.input.weight.new_zeros((batch_size, self.d_h))
        # END ANSWER custom_h0


def predict(rnn, xs, baseline, output_scale):
    """Normalized history + CV forecast -> future relative positions in meters.

    xs: (B,O,4); baseline: (B,H,2); output_scale: a fixed 10-meter scale.
    No future ground-truth values are available to this function.
    """
    # BEGIN ANSWER predict
    h0 = rnn.initial_recurrent_state(xs.shape[0])
    hs = scan(rnn.f, h0, xs)
    residual = rnn.g(hs[:, -1]).reshape_as(baseline)
    return baseline + output_scale * residual
    # END ANSWER predict


def train_one_epoch(rnn, loader, optimizer, device, output_scale):
    """One epoch; return sample-weighted normalized coordinate MSE."""
    # BEGIN ANSWER train
    rnn.train()
    total_loss, count = 0.0, 0
    for xs, baseline, target in loader:
        xs, baseline, target = (a.to(device) for a in (xs, baseline, target))
        optimizer.zero_grad(set_to_none=True)
        prediction = predict(rnn, xs, baseline, output_scale)
        loss = ((prediction - target) / output_scale).square().mean()
        loss.backward()
        nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * xs.shape[0]
        count += xs.shape[0]
    return total_loss / count
    # END ANSWER train


def displacement_errors(prediction, target):
    """Return per-window ADE, FDE: two tensors of shape (B,), in meters."""
    # BEGIN ANSWER metrics
    distance = torch.linalg.vector_norm(prediction - target, dim=-1)
    return distance.mean(dim=-1), distance[:, -1]
    # END ANSWER metrics


if __name__ == "__main__":
    run_experiment(
        {"elman": ElmanRNN, "gated": MyRNN}, make_features,
        predict, train_one_epoch, displacement_errors,
    )
