"""Download a small, fixed PHUMA G1 walking subset without the 3.4 GB archive.

The original NPY dictionaries are read with a restricted NumPy-only unpickler.
The resulting NPZ contains only numeric/string arrays (allow_pickle=False).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import pickle
import re
import struct
import time
from urllib.request import Request, urlopen
import zipfile
import zlib

import numpy as np
try:
    from numpy import _core as numpy_core
except ImportError:
    from numpy import core as numpy_core

REVISION = "48b0f5982969104ab041461e594a8b8c36db8ba5"
URL = f"https://huggingface.co/datasets/DAVIAN-Robotics/PHUMA/resolve/{REVISION}/data.zip"
ROOT = Path(__file__).resolve().parent
PATTERN = r"number_|hand_pockets|arm_swing|clapping|salute|waving|pointing"


class RemoteZip(io.RawIOBase):
    def __init__(self):
        self.pos = 0
        with urlopen(Request(URL, method="HEAD"), timeout=90) as response:
            self.url = response.url
            self.size = int(response.headers["Content-Length"])

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def fetch(self, start, count):
        if count == 0:
            return b""
        for attempt in range(4):
            try:
                request = Request(self.url, headers={"Range": f"bytes={start}-{start + count - 1}"})
                with urlopen(request, timeout=120) as response:
                    if response.status != 206:
                        raise ValueError("Server must support HTTP Range; refusing full archive download.")
                    raw = response.read(count + 1)
                if len(raw) != count:
                    raise ValueError("Truncated or oversized HTTP range.")
                return raw
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1 + attempt)

    def read(self, count=-1):
        count = self.size - self.pos if count < 0 else min(count, self.size - self.pos)
        raw = self.fetch(self.pos, count)
        self.pos += count
        return raw


class NumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
        }
        for prefix in ("numpy.core.multiarray", "numpy._core.multiarray"):
            allowed[prefix, "_reconstruct"] = numpy_core.multiarray._reconstruct
            allowed[prefix, "scalar"] = numpy_core.multiarray.scalar
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported global: {module}.{name}")
        return allowed[module, name]


def read_original(raw):
    stream = io.BytesIO(raw)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"Unsupported NPY version: {version}")
    if shape != () or not dtype.hasobject:
        raise ValueError("Expected the official scalar dictionary NPY format.")
    return NumpyUnpickler(stream).load().item()


def selected(name):
    return (name.startswith("data/g1/idea400/") and name.endswith(".npy")
            and "walking" in name.lower() and re.search(PATTERN, name, re.I) is not None)


def source_group(name):
    # Keep all clips/chunks of the same named source recording in one split.
    return re.split(r"_clip_?\d+|_chunk_", name.removeprefix("data/g1/"))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "raw_cache")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    remote = RemoteZip()
    with zipfile.ZipFile(remote) as archive:
        infos = sorted((i for i in archive.infolist() if selected(i.filename)), key=lambda i: i.filename)
    if not infos:
        raise ValueError("No matching walking motions found.")
    print(f"Selected {len(infos)} files from pinned PHUMA release.", flush=True)

    def download(info):
        cache = args.cache / Path(info.filename).name
        if cache.exists():
            raw = cache.read_bytes()
        else:
            # A small over-read covers the variable-length local ZIP header.
            block = remote.fetch(info.header_offset, 30 + len(info.filename.encode()) + 256 + info.compress_size)
            fields = struct.unpack("<4s5H3I2H", block[:30])
            if fields[0] != b"PK\x03\x04" or fields[3] != zipfile.ZIP_DEFLATED:
                raise ValueError("Unexpected local ZIP header/compression.")
            start = 30 + fields[-2] + fields[-1]
            if start + info.compress_size > len(block):
                block = remote.fetch(info.header_offset, start + info.compress_size)
            if block[30:30 + fields[-2]].decode() != info.filename:
                raise ValueError("ZIP filename mismatch.")
            raw = zlib.decompress(block[start:start + info.compress_size], -15)
        if len(raw) != info.file_size or zlib.crc32(raw) != info.CRC:
            raise ValueError(f"ZIP integrity check failed: {info.filename}")
        cache.write_bytes(raw)
        motion = read_original(raw)
        xyz = np.asarray(motion["root_trans"], dtype=np.float32)
        fps = float(motion["fps"])
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all() or fps != 30:
            raise ValueError(f"Unexpected coordinates/frame rate: {info.filename}")
        return xyz, {"archive_path": info.filename, "source_group": source_group(info.filename),
                     "frames_raw": len(xyz), "fps_raw": fps, "sha256_raw": hashlib.sha256(raw).hexdigest(),
                     "zip_crc32": info.CRC}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(download, infos))
    # Pure decimation: no smoothing across the observation/future boundary.
    arrays = [xyz[::3] for xyz, _ in results]
    offsets = np.concatenate(([0], np.cumsum([len(a) for a in arrays])))
    path = args.output / "phuma_g1_walking.npz"
    np.savez_compressed(path, root_xyz=np.concatenate(arrays), offsets=offsets,
                        names=np.array([m["archive_path"] for _, m in results]),
                        groups=np.array([m["source_group"] for _, m in results]), fps=np.array(10.0))
    manifest = {
        "dataset": "PHUMA", "robot": "Unitree G1", "revision": REVISION, "url": URL,
        "retrieved_date": time.strftime("%Y-%m-%d"), "license_declared_by_dataset_card": "Apache-2.0",
        "selection": {"prefix": "data/g1/idea400/", "contains": "walking", "regex": PATTERN},
        "task": "2D root translation forecast for walking with simple upper-body gestures",
        "data_kind": "Retargeted humanoid reference motion; not measured robot execution logs",
        "coordinates": "Z-up source world frame, meters; model uses root_trans[:, :2]",
        "processing": "Take every third 30 Hz frame to 10 Hz; retain xyz; no smoothing or interpolation",
        "split_policy": "Group by source name before clip/chunk suffix; seed 42, 70/15/15; classroom split, not paper benchmark",
        "sha256_npz": hashlib.sha256(path.read_bytes()).hexdigest(),
        "num_sequences": len(arrays), "num_source_groups": len(set(m["source_group"] for _, m in results)),
        "frames_10hz": int(offsets[-1]), "sequences": [m for _, m in results],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("num_sequences", "num_source_groups", "frames_10hz", "sha256_npz")}), flush=True)


if __name__ == "__main__":
    main()
