"""Public checks: python check_assignment.py --module student [--model elman]."""

import argparse
import importlib

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trajectory_utils import DATA, DT, constant_velocity, read_tracks, split_tracks


def check(module, model_name):
    torch.manual_seed(9)
    torch.set_num_threads(2)
    times = torch.arange(15) * DT
    history = torch.stack((2 * times, 4 * times), dim=-1)[None].repeat(3, 1, 1)
    features = module.make_features(history, DT)
    assert features.shape == (3, 15, 4)
    assert torch.isfinite(features).all()
    torch.testing.assert_close(module.make_features(history + 20, DT), features, atol=3e-5, rtol=1e-5)
    torch.testing.assert_close(features[:, -1, :2], torch.zeros(3, 2))
    torch.testing.assert_close(features[0, :, 2:], torch.tensor([2.0, 4.0]).expand(15, 2))
    baseline = constant_velocity(history, 15)
    future_times = torch.arange(1, 16) * DT
    exact = torch.stack((2 * future_times, 4 * future_times), dim=-1)
    torch.testing.assert_close(baseline, exact[None].expand(3, 15, 2))
    target = torch.zeros(3, 15, 2)
    shifted = target + torch.tensor([3.0, 4.0])
    ade, fde = module.displacement_errors(shifted, target)
    torch.testing.assert_close(ade, torch.full((3,), 5.0))
    torch.testing.assert_close(fde, torch.full((3,), 5.0))
    shifted.zero_()
    shifted[:, -1, 0] = 15
    ade, fde = module.displacement_errors(shifted, target)
    torch.testing.assert_close(ade, torch.ones(3))
    torch.testing.assert_close(fde, torch.full((3,), 15.0))
    cls = module.ElmanRNN if model_name == "elman" else module.MyRNN
    model = cls(4, 12, 30)
    assert "forward" not in cls.__dict__, "Use f, g, scan as in the assignment."
    forbidden = (nn.RNN, nn.RNNCell, nn.GRU, nn.GRUCell, nn.LSTM, nn.LSTMCell)
    assert not any(isinstance(layer, forbidden) for layer in model.modules())
    xs = torch.randn(3, 15, 4, requires_grad=True)
    hs = module.scan(model.f, model.initial_recurrent_state(3), xs)
    assert hs.shape == (3, 15, 12)
    gradient = torch.autograd.grad(hs[:, -1].square().sum(), xs)[0]
    assert gradient[:, 0].abs().sum() > 0, "scan detached the recurrent history."
    pred = module.predict(model, xs, baseline, 10.0)
    assert pred.shape == (3, 15, 2) and torch.isfinite(pred).all()
    for i in range(3):
        torch.testing.assert_close(pred[i:i + 1], module.predict(model, xs[i:i + 1], baseline[i:i + 1], 10.0))
    # A tiny learnable residual task catches a missing backward/optimizer step.
    learning_target = baseline + 0.5
    loader = DataLoader(TensorDataset(xs.detach(), baseline, learning_target), batch_size=3)
    before = ((pred.detach() - learning_target) ** 2).mean().item()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(25):
        loss = module.train_one_epoch(model, loader, optimizer, "cpu", 10.0)
        assert isinstance(loss, (float, int)) and loss >= 0
    after = ((module.predict(model, xs, baseline, 10.0) - learning_target) ** 2).mean().item()
    assert after < before * 0.3, f"Training did not fit the tiny task: {before} -> {after}"
    tracks, _ = read_tracks(DATA)
    splits = split_tracks(tracks, 42, 30)
    assert len(set(sum(splits.values(), []))) == sum(map(len, splits.values()))
    print(f"PASS: {model_name}: shapes, units, metrics, history gradients, independent batches, training, splits")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", choices=["student", "solution"], default="student")
    parser.add_argument("--model", choices=["both", "elman", "gated"], default="both")
    args = parser.parse_args()
    module = importlib.import_module(args.module)
    for name in (["elman", "gated"] if args.model == "both" else [args.model]):
        check(module, name)
