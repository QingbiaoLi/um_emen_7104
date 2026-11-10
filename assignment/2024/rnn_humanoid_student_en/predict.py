"""Forecast an (O,2) or (B,O,2) .npy history in meters, sampled every 0.1 s.

python predict.py --module solution --checkpoint outputs/reference/gated_best.pt \
    --history history.npy --output forecast.npy
"""

import argparse
import importlib
from pathlib import Path

import numpy as np
import torch

from trajectory_utils import constant_velocity


@torch.no_grad()
def forecast(checkpoint_path, observed, module_name="student"):
    module = importlib.import_module(module_name)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = saved["config"]
    observed = torch.as_tensor(observed, dtype=torch.float32)
    single = observed.ndim == 2
    if single:
        observed = observed.unsqueeze(0)
    if observed.ndim != 3 or tuple(observed.shape[1:]) != (config["obs_len"], 2):
        raise ValueError(f"Expected (B,{config['obs_len']},2) or ({config['obs_len']},2).")
    if not torch.isfinite(observed).all():
        raise ValueError("History contains NaN or infinity.")
    model_class = {"elman": module.ElmanRNN, "gated": module.MyRNN}[saved["model"]]
    model = model_class(4, config["hidden"], config["pred_len"] * 2)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    features = (module.make_features(observed, saved["dt"]) - saved["mean"]) / saved["std"]
    baseline = constant_velocity(observed, config["pred_len"], saved["dt"])
    relative = module.predict(model, features, baseline, saved["output_scale"])
    absolute = (relative + observed[:, -1:]).numpy()
    return absolute[0] if single else absolute


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", choices=["solution", "student"], default="student")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("forecast.npy"))
    args = parser.parse_args()
    result = forecast(args.checkpoint, np.load(args.history, allow_pickle=False), args.module)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, result)
    print(f"Saved {result.shape} absolute positions in meters to {args.output}")
