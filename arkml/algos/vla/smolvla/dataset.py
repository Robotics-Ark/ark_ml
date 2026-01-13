import os
import pickle
from collections import OrderedDict
from threading import Lock
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from arkml.core.app_context import ArkMLContext
from arkml.utils.utils import _image_to_tensor
from torch.utils.data import Dataset
from torchvision import transforms


class smolVLADataset(Dataset):
    def __init__(
        self,
        dataset_path,
        transform=None,
        pred_horizon: int = 1,
        image_base_index: int = 9,
        # Caching controls
        cache: str | None = "all",  # 'file', 'all'
        # Maximum number of pickle files to keep in memory when using file cache.
        # Set to None for unbounded (may use more RAM). Ignored when cache == "all".
        max_cached_files: int | None = 16,
        *args,
        **kwargs,
    ):
        self.pred_horizon = pred_horizon

        super().__init__()
        self.dataset_path = dataset_path
        self.transform = transform or transforms.ToTensor()
        self.image_base_index = image_base_index

        self.index_map = []
        # cache options: None/"none" (no cache), "file" (LRU per-file cache), "all" (preload all files)
        self.cache_mode = (cache or "none").lower()
        if self.cache_mode not in {"none", "file", "all"}:
            raise ValueError(f"Unknown cache mode: {self.cache_mode}")
        self.max_cached_files = max_cached_files

        # Per-process (worker) cache structures
        self._cache_lock: Lock = Lock()
        # LRU of file_path -> traj_list
        self._file_cache: "OrderedDict[str, List[dict]]" = OrderedDict()

        self._build_index_map()
        if self.cache_mode == "all":
            self._preload_all_files()

    """Lazy-loading dataset that adapts to configurable visual inputs."""

    def _build_index_map(self) -> None:
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"Dataset path '{self.dataset_path}' does not exist."
            )

        file_list = sorted(
            [
                os.path.join(self.dataset_path, f)
                for f in os.listdir(self.dataset_path)
                if f.endswith(".pkl")
            ]
        )
        file_list=file_list[:2]

        for fpath in file_list:
            with open(fpath, "rb") as f:
                traj_list = pickle.load(f)
                for traj_idx, traj in enumerate(traj_list):
                    actions = np.asarray(traj["action"], dtype=np.float32)
                    if actions.size == 0:
                        continue
                    if actions.size == 1:
                        actions = actions[None, :]

                    num_steps = actions.shape[0]

                    for step_idx in range(num_steps):
                        self.index_map.append((fpath, traj_idx, step_idx))

    def _preload_all_files(self) -> None:
        """Preload every pickle file referenced by the index into RAM.

        This happens per DataLoader worker process (safe). Useful for maximum
        throughput at the cost of memory. No-op if cache_mode != 'all'.
        """
        if self.cache_mode != "all":
            return
        # Collect unique file paths from index_map
        unique_files = sorted({f for f, _, _ in self.index_map})
        for fpath in unique_files:
            # Load once and insert into cache
            with open(fpath, "rb") as f:
                traj_list = pickle.load(f)
            with self._cache_lock:
                self._file_cache[fpath] = traj_list

    def _get_traj_list(self, fpath: str) -> List[dict]:
        """Return trajectory list for file path, using cache if enabled."""
        if self.cache_mode == "none":
            with open(fpath, "rb") as f:
                return pickle.load(f)

        # file or all modes use the cache
        with self._cache_lock:
            cached = self._file_cache.get(fpath)
            if cached is not None:
                # Move to end to mark as recently used
                self._file_cache.move_to_end(fpath)
                return cached

        # Not in cache: load from disk
        with open(fpath, "rb") as f:
            traj_list = pickle.load(f)

        # Insert into cache with LRU eviction for 'file' mode
        with self._cache_lock:
            self._file_cache[fpath] = traj_list
            self._file_cache.move_to_end(fpath)
            if self.cache_mode == "file" and self.max_cached_files is not None:
                while len(self._file_cache) > self.max_cached_files:
                    self._file_cache.popitem(last=False)
        return traj_list

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx) -> dict[str, Any]:
        fpath, traj_idx, step_index = self.index_map[idx]
        traj_list = self._get_traj_list(fpath)
        trajectory = traj_list[traj_idx]

        sample: dict[str, Any] = {"task": "pick and place the object"}

        state_array = np.asarray(
            trajectory["state"][6], dtype=np.float32
        )  # TODO handle proper index based on data collection pipeline
        sample["state"] = torch.from_numpy(state_array)

        for cam_index, cam_name in enumerate(ArkMLContext.visual_input_features):
            image_value = trajectory.get(cam_name)
            if image_value is None:
                state_block = trajectory.get("state")
                if state_block is not None:
                    candidate_idx = self.image_base_index + cam_index
                    if len(state_block) > candidate_idx:
                        image_value = state_block[candidate_idx]
            if image_value is None:
                raise KeyError(f"Image data for '{cam_name}' not found in trajectory")
            sample[cam_name] = _image_to_tensor(
                image_value=image_value, transform=self.transform
            )

        action_array = np.asarray(trajectory["action"], dtype=np.float32)
        if action_array.ndim == 1:
            action_array = action_array[None, :]

        action_window = action_array[step_index : step_index + self.pred_horizon]
        horizon = action_window.shape[0]
        padded_actions = np.zeros(
            (self.pred_horizon, action_array.shape[1]), dtype=np.float32
        )
        padded_actions[:horizon] = action_window

        action_is_pad = np.ones(self.pred_horizon, dtype=bool)
        action_is_pad[:horizon] = False

        sample["action"] = torch.from_numpy(padded_actions)
        sample["action_is_pad"] = torch.from_numpy(action_is_pad)

        return sample
