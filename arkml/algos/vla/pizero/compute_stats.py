import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image


def estimate_num_samples(
    dataset_len: int, min_num_samples: int = 100, max_num_samples: int = 10_000, power: float = 0.75
) -> int:
    """Heuristic from LeRobot to estimate sample count based on dataset size."""
    if dataset_len < min_num_samples:
        min_num_samples = dataset_len
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))


def sample_indices(data_len: int) -> list[int]:
    num_samples = estimate_num_samples(data_len)
    if data_len <= 1:
        return [0]
    return np.round(np.linspace(0, data_len - 1, num_samples)).astype(int).tolist()


def _accumulate_moments(x: np.ndarray, state: Dict[str, Any]) -> None:
    """Accumulate count, sum, sumsq, min, max across the first axis of x.

    x is expected to be shape (N, D...) where axis=0 is the sample axis.
    """
    x = np.asarray(x)
    n = x.shape[0]
    state["count"] += n
    state["sum"] += x.sum(axis=0)
    state["sumsq"] += np.square(x).sum(axis=0)
    state["min"] = np.minimum(state["min"], x.min(axis=0))
    state["max"] = np.maximum(state["max"], x.max(axis=0))


def _finalize_stats(state: Dict[str, Any]) -> Dict[str, np.ndarray]:
    count = max(1, int(state["count"]))
    mean = state["sum"] / count
    var = np.maximum(0.0, state["sumsq"] / count - np.square(mean))
    std = np.sqrt(var)
    return {
        "min": state["min"],
        "max": state["max"],
        "mean": mean,
        "std": std,
        "count": np.array([count], dtype=np.int64),
    }


def _init_state(shape: Tuple[int, ...], dtype=np.float64) -> Dict[str, Any]:
    return {
        "count": 0,
        "sum": np.zeros(shape, dtype=dtype),
        "sumsq": np.zeros(shape, dtype=dtype),
        "min": np.full(shape, np.inf, dtype=dtype),
        "max": np.full(shape, -np.inf, dtype=dtype),
    }


def _iter_trajectories(dataset_path: str):
    files = sorted([os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".pkl")])
    for fpath in files:
        with open(fpath, "rb") as f:
            traj_list = pickle.load(f)
            for traj in traj_list:
                yield traj


def compute_pizero_stats(dataset_path: str, sample_images_only: bool = True) -> Dict[str, Dict[str, Any]]:
    """Compute dataset statistics for PiZero datasets.

    Assumptions based on current PiZeroDataset:
    - `trajectory["state"][:10]` are numeric states -> shape (T=10, state_dim)
    - `trajectory["state"][10]` is an RGB image array (H, W, C), uint8
    - `trajectory["action"]` is a 1D or 2D array -> coerced to (action_dim,)

    Returns a dict keyed by LeRobot policy feature names:
      - "observation.state": mean/std/min/max with shape (state_dim,)
      - "observation.images.image": mean/std/min/max with shape (3,1,1) in [0,1]
      - "action": mean/std/min/max with shape (action_dim,)
    """
    # First, collect quick shapes to initialize accumulators and dataset length
    trajs = list(_iter_trajectories(dataset_path))
    if len(trajs) == 0:
        raise FileNotFoundError(f"No trajectories found in {dataset_path}")

    # Infer shapes from the first trajectory
    first = trajs[0]
    state_block = np.asarray(first["state"][:10], dtype=np.float64)  # (10, state_dim)
    img_arr = np.asarray(first["state"][10])  # (H,W,C), uint8 assumed
    if img_arr.ndim != 3 or img_arr.shape[-1] != 3:
        raise ValueError("Expected RGB image at trajectory['state'][10] with shape (H,W,3)")
    action_arr = np.asarray(first["action"])  # (action_dim,) or (1, action_dim)
    if action_arr.ndim == 2 and action_arr.shape[0] == 1:
        action_arr = action_arr[0]
    if action_arr.ndim != 1:
        raise ValueError("Expected action to be 1D or (1, D)")

    state_dim = state_block.shape[-1]
    action_dim = action_arr.shape[-1]

    # Accumulators
    s_state = _init_state((state_dim,), dtype=np.float64)
    s_action = _init_state((action_dim,), dtype=np.float64)
    # For images we accumulate per-channel moments; spatial dims are averaged implicitly by flattening
    s_image = _init_state((3,), dtype=np.float64)

    # Decide image sampling indices to keep it light
    num_trajs = len(trajs)
    img_indices = set(sample_indices(num_trajs))

    for i, traj in enumerate(trajs):
        # state: flatten time into batch
        sb = np.asarray(traj["state"][:10], dtype=np.float64)  # (10, state_dim)
        _accumulate_moments(sb, s_state)

        # action
        act = np.asarray(traj["action"])  # (D,) or (1,D)
        if act.ndim == 2 and act.shape[0] == 1:
            act = act[0]
        act = act.astype(np.float64)
        _accumulate_moments(act[None, ...], s_action)

        # image (sampled)
        if (not sample_images_only) or (i in img_indices):
            img = np.asarray(traj["state"][10])  # (H,W,C), uint8
            if img.dtype != np.float32 and img.dtype != np.float64:
                img = img.astype(np.float32)
            img = img / 255.0  # to [0,1]
            # channel-wise mean/std computed over all pixels
            c_means = img.reshape(-1, 3).mean(axis=0)  # (3,)
            c_sumsq = (img.reshape(-1, 3) ** 2).mean(axis=0)  # (3,), will rescale below
            # Convert means over pixels to sum/sumsq with a pseudo-count = num_pixels
            num_pixels = img.shape[0] * img.shape[1]
            # Build a temporary batch of size 1 with channel vector repeated to fit accumulator API
            # We can accumulate directly by updating state fields
            s_image["count"] += num_pixels
            s_image["sum"] += c_means * num_pixels
            s_image["sumsq"] += c_sumsq * num_pixels
            s_image["min"] = np.minimum(s_image["min"], img.reshape(-1, 3).min(axis=0))
            s_image["max"] = np.maximum(s_image["max"], img.reshape(-1, 3).max(axis=0))

    # Finalize stats
    state_stats = _finalize_stats(s_state)
    action_stats = _finalize_stats(s_action)
    image_stats = _finalize_stats(s_image)

    # Match LeRobot shapes: visuals (C,1,1)
    for k in ("mean", "std", "min", "max"):
        image_stats[k] = image_stats[k].reshape(3, 1, 1)

    return {
        "observation.state": state_stats,
        "observation.images.image": image_stats,
        "action": action_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute PiZero dataset stats (LeRobot-compatible)")
    parser.add_argument("--dataset_path", type=str, required=True, help="Directory with .pkl trajectory files")
    parser.add_argument("--out", type=str, required=True, help="Output JSON path for stats")
    parser.add_argument(
        "--full_images",
        action="store_true",
        help="Process images for every trajectory instead of sampling (slower)",
    )
    args = parser.parse_args()

    stats = compute_pizero_stats(args.dataset_path, sample_images_only=not args.full_images)

    # Convert numpy arrays to lists for JSON
    serializable = {k: {kk: vv.tolist() for kk, vv in d.items()} for k, d in stats.items()}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"Wrote stats to {out_path}")


if __name__ == "__main__":
    main()

