"""Provided infrastructure shared by the instructor and student versions."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import random
import time
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "ngsim_i80.csv.gz"
DT = 0.2
OUTPUT_SCALE = 10.0
QUERY = {
    "$select": "distinct vehicle_id,frame_id,global_time,local_x,local_y",
    "$where": "location = 'i-80' AND global_time >= 1113433135000 AND "
              "global_time < 1113434035000 AND vehicle_id <= 500 AND v_class = 2",
    "$order": "vehicle_id,global_time,frame_id",
    "$limit": "500000",
}
DATA_URL = "https://data.transportation.gov/resource/8ect-6jqj.csv?" + urlencode(QUERY)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def read_tracks(path):
    """Validate the fixed single-recording subset; split at any missing frame."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading official NGSIM subset ...", flush=True)
        try:
            with urlopen(DATA_URL, timeout=180) as response:
                raw = response.read()
        except Exception as exc:
            raise RuntimeError("Download failed. Use the bundled data/ngsim_i80.csv.gz "
                               "or download DATA_URL on a connected machine.") from exc
        if not raw.startswith(b'"vehicle_id"') and not raw.startswith(b"vehicle_id"):
            raise ValueError("The endpoint did not return the expected CSV.")
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(gzip.compress(raw, mtime=0))
        temporary.replace(path)
    blob = path.read_bytes()
    raw = gzip.decompress(blob) if path.suffix == ".gz" else blob
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    if len(rows) >= int(QUERY["$limit"]):
        raise ValueError("Query limit reached; refuse a possibly truncated dataset.")
    grouped = {}
    for row in rows:
        vehicle = int(row["vehicle_id"])
        frame = int(row["frame_id"])
        timestamp = int(row["global_time"])
        position = np.array([float(row["local_x"]), float(row["local_y"])]) * 0.3048
        if not np.isfinite(position).all():
            raise ValueError("Non-finite position.")
        if not 1113433135000 <= timestamp < 1113434035000:
            raise ValueError("Only the documented single recording is supported.")
        grouped.setdefault(vehicle, []).append((timestamp, frame, position))
    tracks = {}
    for vehicle, records in sorted(grouped.items()):
        records.sort(key=lambda r: (r[0], r[1]))
        segments, current, previous = [], [], None
        for timestamp, frame, position in records:
            if previous is not None:
                if timestamp == previous[0]:
                    if frame != previous[1] or not np.array_equal(position, previous[2]):
                        raise ValueError(f"Conflicting duplicate for vehicle {vehicle}.")
                    continue
                if timestamp - previous[0] != 100 or frame - previous[1] != 1:
                    if current:
                        segments.append(np.array(current, dtype=np.float32)[::2])
                    current = []
            current.append(position)
            previous = timestamp, frame, position
        if current:
            segments.append(np.array(current, dtype=np.float32)[::2])
        tracks[vehicle] = segments
    return tracks, {"sha256_gzip": hashlib.sha256(blob).hexdigest(),
                    "sha256_csv": hashlib.sha256(raw).hexdigest(), "raw_rows": len(rows)}


def split_tracks(tracks, seed, minimum_length):
    ids = np.array([i for i in sorted(tracks)
                    if any(len(segment) >= minimum_length for segment in tracks[i])])
    if len(ids) < 15:
        raise ValueError("At least 15 sufficiently long vehicles are required.")
    np.random.default_rng(seed).shuffle(ids)
    n_train, n_val = int(0.7 * len(ids)), int(0.15 * len(ids))
    splits = {"train": ids[:n_train], "val": ids[n_train:n_train + n_val],
              "test": ids[n_train + n_val:]}
    assert not (set(splits["train"]) & set(splits["test"]))
    assert not (set(splits["train"]) & set(splits["val"]))
    assert not (set(splits["val"]) & set(splits["test"]))
    return {name: sorted(map(int, values)) for name, values in splits.items()}


def make_windows(tracks, ids, obs_len, pred_len, stride):
    observed, future, vehicle_ids = [], [], []
    for vehicle in ids:
        for segment in tracks[vehicle]:
            for start in range(0, len(segment) - obs_len - pred_len + 1, stride):
                observed.append(segment[start:start + obs_len])
                future.append(segment[start + obs_len:start + obs_len + pred_len])
                vehicle_ids.append(vehicle)
    if not observed:
        raise ValueError("Split contains no complete windows.")
    return torch.from_numpy(np.stack(observed)), torch.from_numpy(np.stack(future)), vehicle_ids


def constant_velocity(observed, pred_len, dt=DT):
    """OLS velocity over the last 1 second; anchor prediction at last observation."""
    count = min(observed.shape[1], round(1.0 / dt) + 1)
    times = torch.arange(count, device=observed.device, dtype=observed.dtype) * dt
    times = times - times.mean()
    velocity = (observed[:, -count:] * times[None, :, None]).sum(1) / times.square().sum()
    horizon = torch.arange(1, pred_len + 1, device=observed.device,
                           dtype=observed.dtype) * dt
    return velocity[:, None] * horizon[None, :, None]


@torch.no_grad()
def evaluate(model, loader, predict, errors, device):
    model.eval()
    predictions, ade, fde = [], [], []
    for xs, baseline, target in loader:
        prediction = predict(model, xs.to(device), baseline.to(device), OUTPUT_SCALE).cpu()
        a, f = errors(prediction, target)
        predictions.append(prediction)
        ade.append(a)
        fde.append(f)
    return {"ADE_m": torch.cat(ade).mean().item(), "FDE_m": torch.cat(fde).mean().item()}, torch.cat(predictions)


def plot_results(out, histories, observed, target, predictions, baseline, vehicle_ids):
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(out / ".cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, history in histories.items():
        ax.plot(range(1, len(history) + 1), [x["val_ADE_m"] for x in history], label=name)
    ax.set(xlabel="Epoch", ylabel="Validation ADE (m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=160)
    plt.close(fig)
    # Fixed indices chosen without looking at prediction errors.
    indices = np.linspace(0, len(target) - 1, 6, dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, index in zip(axes.flat, indices):
        obs = observed[index] - observed[index, -1:]
        ax.plot(obs[:, 0], obs[:, 1], "o-", ms=2, color="0.5", label="History")
        for label, values, style in [("Truth", target, "k-"), ("CV", baseline, "--")]:
            points = torch.cat((torch.zeros(1, 2), values[index]))
            ax.plot(points[:, 0], points[:, 1], style, label=label)
        for name, values in predictions.items():
            points = torch.cat((torch.zeros(1, 2), values[index]))
            ax.plot(points[:, 0], points[:, 1], label=name)
        ax.set(title=f"Vehicle {vehicle_ids[index]}, window {index}",
               xlabel="Lateral displacement (m)", ylabel="Longitudinal displacement (m)")
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"I-80 held-out vehicles: {target.shape[1] * DT:g}-second forecast "
                 "(lateral and longitudinal axes use different scales)")
    fig.tight_layout()
    fig.savefig(out / "trajectories.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, forecast in {"constant velocity": baseline, **predictions}.items():
        error = torch.linalg.vector_norm(forecast - target, dim=-1).mean(0)
        ax.plot(np.arange(1, len(error) + 1) * DT, error, label=name)
    ax.set(xlabel="Time after last observation (s)", ylabel="Mean displacement error (m)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "error_by_horizon.png", dpi=160)
    plt.close(fig)


def run_experiment(model_classes, feature_fn, predict, train_one_epoch, errors):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--obs-len", type=int, default=15)
    parser.add_argument("--pred-len", type=int, default=15)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--model", choices=["both", "elman", "gated"], default="both")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.hidden, args.pred_len, args.stride, args.threads) < 1 or args.obs_len < 2 or args.lr <= 0:
        parser.error("Lengths/counts/lr must be positive, and obs-len must be >= 2.")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    tracks, provenance = read_tracks(args.data)
    splits = split_tracks(tracks, args.seed, args.obs_len + args.pred_len)
    windows = {s: make_windows(tracks, ids, args.obs_len, args.pred_len, args.stride)
               for s, ids in splits.items()}
    features = {s: feature_fn(w[0], DT) for s, w in windows.items()}
    mean = features["train"].mean(dim=(0, 1), keepdim=True)
    std = features["train"].std(dim=(0, 1), keepdim=True).clamp_min(1e-3)
    datasets = {}
    for s, (observed, future, _) in windows.items():
        datasets[s] = TensorDataset((features[s] - mean) / std,
                                    constant_velocity(observed, args.pred_len),
                                    future - observed[:, -1:])
    counts = {s: {"vehicles": len(splits[s]), "windows": len(datasets[s])} for s in splits}
    print(json.dumps(counts, indent=2), flush=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    report = {"config": config, "dt_seconds": DT, "counts": counts,
              "data": {**provenance, "url": DATA_URL}, "split_vehicle_ids": splits,
              "versions": {"torch": str(torch.__version__), "numpy": np.__version__},
              "test": {}, "models": {}}
    (args.output / "splits.json").write_text(json.dumps(splits, indent=2))
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size)
    histories, models = {}, {}
    for name, model_class in model_classes.items():
        if args.model not in ("both", name):
            continue
        seed_everything(args.seed)
        loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(args.seed))
        model = model_class(4, args.hidden, args.pred_len * 2).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        best, history, started = float("inf"), [], time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            loss = train_one_epoch(model, loader, optimizer, args.device, OUTPUT_SCALE)
            metric, _ = evaluate(model, val_loader, predict, errors, args.device)
            if not np.isfinite(loss) or not np.isfinite(metric["ADE_m"]):
                raise RuntimeError("Non-finite training/validation metric.")
            history.append({"epoch": epoch, "train_normalized_MSE": loss,
                            "val_ADE_m": metric["ADE_m"], "val_FDE_m": metric["FDE_m"]})
            if metric["ADE_m"] < best:
                best = metric["ADE_m"]
                checkpoint = {"model": name, "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                              "mean": mean, "std": std, "config": config, "dt": DT,
                              "output_scale": OUTPUT_SCALE, "epoch": epoch,
                              "val_ADE_m": best, "data_sha256": provenance["sha256_csv"]}
                torch.save(checkpoint, args.output / f"{name}_best.pt")
            if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                print(f"{name:5} {epoch:3}/{args.epochs}: loss={loss:.5f}, "
                      f"val ADE={metric['ADE_m']:.3f} m, FDE={metric['FDE_m']:.3f} m", flush=True)
        model.load_state_dict(checkpoint["state_dict"])
        histories[name], models[name] = history, model
        report["models"][name] = {"best_epoch": checkpoint["epoch"], "val_ADE_m": best,
                                  "training_seconds": time.perf_counter() - started}
    # Test targets are used for final reporting only, after all model selection.
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size)
    _, baseline, target = datasets["test"].tensors
    for name, forecast in (("constant_velocity", baseline), ("last_position", torch.zeros_like(target))):
        ade, fde = errors(forecast, target)
        report["test"][name] = {"ADE_m": ade.mean().item(), "FDE_m": fde.mean().item()}
    predictions = {}
    for name, model in models.items():
        report["test"][name], predictions[name] = evaluate(model, test_loader, predict, errors, args.device)
    report["history"] = histories
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(args.output / "test_predictions.npz", observed=windows["test"][0].numpy(),
                        target_relative=target.numpy(), baseline_relative=baseline.numpy(),
                        vehicle_ids=np.array(windows["test"][2]),
                        **{k + "_relative": v.numpy() for k, v in predictions.items()})
    plot_results(args.output, histories, windows["test"][0], target, predictions, baseline, windows["test"][2])
    print("Final held-out test metrics (meters):", flush=True)
    print(json.dumps(report["test"], indent=2), flush=True)
    print(f"Artifacts: {args.output.resolve()}", flush=True)
