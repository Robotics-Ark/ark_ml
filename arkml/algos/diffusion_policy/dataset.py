import glob
import numpy as np
import pickle
from pathlib import Path
from typing import List

import torch
from arkml.utils.utils import _image_to_tensor
from torch.utils.data import Dataset
from torchvision import transforms

from arkml.core.app_context import ArkMLContext


class DiffusionPolicyDataset(Dataset):
    def __init__(
        self,
        dataset_path: str | Path,
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        subsample: int = 1,
        in_memory: bool = True,
        transform=None,  # TODO
    ):
        super().__init__()
        self.subsample = subsample

        self.dataset_path = Path(dataset_path)
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.in_memory = in_memory
        self.transform = transform or transforms.Compose(
            [transforms.Resize((256, 256))]
        )

        self.trajectories = []
        self.file_index = sorted(glob.glob(str(self.dataset_path / "*.pkl")))
        if not self.file_index:
            raise FileNotFoundError(f"No *.pkl files in {self.dataset_path}")

        self.trajectories = [None] * len(self.file_index)  # placeholder

        # How many steps are needed for one training sample
        span = (
            self.obs_horizon + self.action_horizon + self.pred_horizon - 1
        ) * self.subsample + 1

        self.sample_index = []
        for tid, traj in enumerate(
            self._get_traj(i) for i in range(len(self.file_index))
        ):
            if len(traj) < span:
                continue  # trajectory too short for even one window
            last_valid_start = len(traj) - span
            self.sample_index.extend([(tid, s) for s in range(last_valid_start + 1)])

    def _get_traj(self, idx: int) -> list[dict]:
        """Return trajectory `idx` (load from disk if necessary)."""
        if self.trajectories[idx] is None:  # lazy-load
            with open(self.file_index[idx], "rb") as f:
                self.trajectories[idx] = pickle.load(f)
        return self.trajectories[idx]

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        traj_id, start = self.sample_index[idx]
        traj = self._get_traj(traj_id)
        step = self.subsample

        # Build index lists
        obs_indices = [start + i * step for i in range(self.obs_horizon)]
        past_indices = [
            start + self.obs_horizon - j * step - 1
            for j in reversed(range(self.action_horizon))
        ]
        pred_indices = [
            start + (self.obs_horizon + i) * step for i in range(self.pred_horizon)
        ]

        # Only the latest observation is used by the model
        last_t = obs_indices[-1]
        latest_state = torch.tensor(traj[last_t]["state"][6][:8], dtype=torch.float32)
        latest_image = self.transform(_image_to_tensor(traj[last_t]["state"][9]))

        # Past actions for transition branch
        past_np = np.asarray(
            [traj[t]["action"] for t in past_indices], dtype=np.float32
        )
        past_actions = torch.from_numpy(past_np)  # (action_horizon, action_dim)

        # Future action window (supervision target)
        pred_np = np.asarray(
            [traj[t]["action"] for t in pred_indices], dtype=np.float32
        )
        pred_actions = torch.from_numpy(pred_np)  # (pred_horizon, action_dim)

        return {
            "state": latest_state,  # (state_dim,)
            ArkMLContext.visual_input_features[0]: latest_image,  # (C,H,W)
            "past_actions": past_actions,  # (action_horizon, action_dim)
            "action": pred_actions,  # (pred_horizon, action_dim)
        }
