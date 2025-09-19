import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .config_utils import resolve_visual_feature_names


def estimate_num_samples(
    dataset_len: int,
    min_num_samples: int = 100,
    max_num_samples: int = 10_000,
    power: float = 0.75,
) -> int:
    if dataset_len < min_num_samples:
        min_num_samples = dataset_len
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))


def sample_indices(data_len: int) -> list[int]:
    num_samples = estimate_num_samples(data_len)
    if data_len <= 1:
        return [0]
    return np.round(np.linspace(0, data_len - 1, num_samples)).astype(int).tolist()


def _accumulate_moments(x: np.ndarray, state: dict[str, Any]) -> None:
    x = np.asarray(x)
    n = x.shape[0]
    state["count"] += n
    state["sum"] += x.sum(axis=0)
    state["sumsq"] += np.square(x).sum(axis=0)
    state["min"] = np.minimum(state["min"], x.min(axis=0))
    state["max"] = np.maximum(state["max"], x.max(axis=0))


def _finalize_stats(state: dict[str, Any]) -> dict[str, np.ndarray]:
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


def _init_state(shape: tuple[int, ...], dtype=np.float64) -> dict[str, Any]:
    return {
        "count": 0,
        "sum": np.zeros(shape, dtype=dtype),
        "sumsq": np.zeros(shape, dtype=dtype),
        "min": np.full(shape, np.inf, dtype=dtype),
        "max": np.full(shape, -np.inf, dtype=dtype),
    }


def _iter_trajectories(dataset_path: str):
    files = sorted(
        [
            os.path.join(dataset_path, f)
            for f in os.listdir(dataset_path)
            if f.endswith(".pkl")
        ]
    )
    for fpath in files:
        with open(fpath, "rb") as f:
            traj_list = pickle.load(f)
            for traj in traj_list:
                yield traj


def compute_pizero_stats(
    dataset_path: str,
    *,
    visual_input_features=None,
    obs_dim: int = 9,
    action_dim: int = 8,
    image_channels: int = 3,
    sample_images_only: bool = True,
    image_base_index: int = 9,
) -> dict[str, dict[str, Any]]:
    camera_names = resolve_visual_feature_names(visual_input_features)

    trajectories = list(_iter_trajectories(dataset_path))
    if not trajectories:
        raise FileNotFoundError(f"No trajectories found in {dataset_path}")

    accumulators: dict[str, dict[str, Any]] = {}
    accumulators["observation.state"] = _init_state((obs_dim,), dtype=np.float64)
    accumulators["action"] = _init_state((action_dim,), dtype=np.float64)
    for cam_name in camera_names:
        key = f"observation.images.{cam_name}"
        accumulators[key] = _init_state((image_channels,), dtype=np.float64)

    sample_idxs = (
        set(sample_indices(len(trajectories))) if sample_images_only else set(range(len(trajectories)))
    )

    for idx, traj in enumerate(trajectories):
        state_block = np.asarray(traj["state"][6], dtype=np.float64)
        _accumulate_moments(state_block.reshape(1, -1), accumulators["observation.state"])

        action = np.asarray(traj["action"], dtype=np.float64)
        if action.ndim == 1:
            action = action.reshape(1, -1)
        _accumulate_moments(action, accumulators["action"])

        if sample_images_only and idx not in sample_idxs:
            continue

        for cam_idx, cam_name in enumerate(camera_names):
            image_value = traj.get(cam_name)
            if image_value is None:
                state_values = traj.get("state")
                if state_values is not None:
                    image_index = image_base_index + cam_idx
                    if len(state_values) > image_index:
                        image_value = state_values[image_index]
            if image_value is None:
                raise KeyError(f"Image data for '{cam_name}' not found in trajectory")

            img = np.asarray(image_value, dtype=np.float64)
            if img.max() > 1.0:
                img = img / 255.0
            channels = img.reshape(-1, image_channels)
            accum_key = f"observation.images.{cam_name}"
            _accumulate_moments(channels, accumulators[accum_key])

    stats = {key: _finalize_stats(acc) for key, acc in accumulators.items()}

    for cam_name in camera_names:
        key = f"observation.images.{cam_name}"
        for stat_key in ("mean", "std", "min", "max"):
            stats[key][stat_key] = stats[key][stat_key].reshape(image_channels, 1, 1)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Compute PiZero dataset stats (configurable cameras)"
    )
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument(
        "--full_images",
        action="store_true",
        help="Process every trajectory image instead of sampling",
    )
    parser.add_argument(
        "--visual_input_features",
        type=str,
        default=None,
        help="Camera list (JSON string) or path to YAML fragment",
    )
    args = parser.parse_args()

    stats = compute_pizero_stats(
        args.dataset_path,
        visual_input_features=args.visual_input_features,
        sample_images_only=not args.full_images,
    )

    serializable = {
        k: {kk: vv.tolist() for kk, vv in d.items()} for k, d in stats.items()
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"Wrote stats to {out_path}")


if __name__ == "__main__":
    main()
