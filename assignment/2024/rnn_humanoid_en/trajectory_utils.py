"""Provided infrastructure shared by the instructor and student versions."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "phuma_g1_walking.npz"
DT = 0.1
OUTPUT_SCALE = 1.0
DATA_URL = "https://huggingface.co/datasets/DAVIAN-Robotics/PHUMA"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def read_tracks(path):
    """Group separate motion chunks by their original source recording."""
    if not path.exists():
        raise FileNotFoundError("Use bundled data or run python prepare_phuma.py first.")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if digest != manifest["sha256_npz"]:
        raise ValueError("Dataset checksum differs from manifest.json.")
    with np.load(path, allow_pickle=False) as data:
        xyz, offsets, groups, names = (data[k] for k in ("root_xyz", "offsets", "groups", "names"))
        if float(data["fps"]) != 1 / DT:
            raise ValueError("Expected 10 Hz positions.")
        if (xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all()
                or offsets.shape != (len(groups) + 1,) or len(names) != len(groups)
                or offsets[0] != 0 or offsets[-1] != len(xyz)
                or not np.all(np.diff(offsets) > 0)):
            raise ValueError("Malformed motion arrays.")
        tracks = {}
        for i, group in enumerate(groups):
            # Never concatenate chunks: there may be temporal gaps between them.
            tracks.setdefault(str(group), []).append(xyz[offsets[i]:offsets[i+1], :2].copy())
    return tracks, {"sha256_npz": digest, "sequences": len(groups), "source_groups": len(tracks),
                    "revision": manifest["revision"], "raw_frames": sum(m["frames_raw"] for m in manifest["sequences"])}


def split_tracks(tracks, seed, minimum_length):
    ids = np.array([i for i in sorted(tracks)
                    if any(len(segment) >= minimum_length for segment in tracks[i])])
    if len(ids) < 15:
        raise ValueError("At least 15 sufficiently long source recordings are required.")
    np.random.default_rng(seed).shuffle(ids)
    n_train, n_val = int(0.7 * len(ids)), int(0.15 * len(ids))
    splits = {"train": ids[:n_train], "val": ids[n_train:n_train + n_val], "test": ids[n_train + n_val:]}
    assert len(set(np.concatenate(list(splits.values())))) == len(ids)
    return {name: sorted(map(str, values)) for name, values in splits.items()}


def make_windows(tracks, ids, obs_len, pred_len, stride):
    observed, future, source_ids = [], [], []
    for source in ids:
        for segment in tracks[source]:
            for start in range(0, len(segment) - obs_len - pred_len + 1, stride):
                observed.append(segment[start:start + obs_len])
                future.append(segment[start + obs_len:start + obs_len + pred_len])
                source_ids.append(source)
    if not observed:
        raise ValueError("Split contains no complete windows.")
    return torch.from_numpy(np.stack(observed)), torch.from_numpy(np.stack(future)), source_ids


def constant_velocity(observed, pred_len, dt=DT):
    """OLS velocity over the last 0.5 seconds; anchor prediction at last observation."""
    count = min(observed.shape[1], round(0.5 / dt) + 1)
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


def plot_results(out, histories, observed, target, predictions, baseline, source_ids):
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
        ax.set(title=f"Test window {index}", xlabel="Root x displacement (m)", ylabel="Root y displacement (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"PHUMA G1 held-out recordings: {target.shape[1] * DT:g}-second root forecast")
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


def run_experiment(model_classes, feature_fn, predict, train_one_epoch, errors, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "my_run")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--obs-len", type=int, default=20)
    parser.add_argument("--pred-len", type=int, default=10)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--model", choices=["both", "elman", "gated"], default="both")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.hidden, args.pred_len, args.stride, args.threads) < 1 or args.obs_len < 2 or args.lr <= 0:
        parser.error("Lengths/counts/lr must be positive, and obs-len must be >= 2.")
    if (args.output / "metrics.json").exists():
        parser.error("Output already contains an experiment; choose a new --output directory.")
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
    counts = {s: {"source_recordings": len(splits[s]), "windows": len(datasets[s])} for s in splits}
    print(json.dumps(counts, indent=2), flush=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    report = {"config": config, "dt_seconds": DT, "counts": counts,
              "data": {**provenance, "url": DATA_URL}, "split_source_ids": splits,
              "versions": {"torch": str(torch.__version__), "numpy": np.__version__},
              "test": {}, "models": {}}
    (args.output / "splits.json").write_text(json.dumps(splits, indent=2))
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size)
    histories, models = {}, {}
    val_cv = errors(datasets["val"].tensors[1], datasets["val"].tensors[2])
    report["validation_constant_velocity"] = {"ADE_m": val_cv[0].mean().item(), "FDE_m": val_cv[1].mean().item()}
    print("Validation CV:", report["validation_constant_velocity"], flush=True)
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
                              "val_ADE_m": best, "data_sha256": provenance["sha256_npz"]}
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
    # Also report equally weighted source means; overlapping windows are correlated.
    report["test_macro_source"] = {}
    source_array = np.array(windows["test"][2])
    for name, forecast in {"constant_velocity": baseline, "last_position": torch.zeros_like(target), **predictions}.items():
        a, f = errors(forecast, target)
        report["test_macro_source"][name] = {
            "ADE_m": float(np.mean([a[source_array == group].mean().item() for group in splits["test"]])),
            "FDE_m": float(np.mean([f[source_array == group].mean().item() for group in splits["test"]])),
        }
    report["history"] = histories
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(args.output / "test_predictions.npz", observed=windows["test"][0].numpy(),
                        target_relative=target.numpy(), baseline_relative=baseline.numpy(),
                        source_ids=np.array(windows["test"][2]),
                        **{k + "_relative": v.numpy() for k, v in predictions.items()})
    plot_results(args.output, histories, windows["test"][0], target, predictions, baseline, windows["test"][2])
    print("Final held-out test metrics (meters):", flush=True)
    print(json.dumps(report["test"], indent=2), flush=True)
    print(f"Artifacts: {args.output.resolve()}", flush=True)
