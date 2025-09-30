import glob
import pickle
from pathlib import Path
from typing import List

import torch
from arkml.utils.utils import _image_to_tensor
from torch.utils.data import Dataset
from torchvision import transforms


class DiffusionPolicyDataset(Dataset):

    def __init__(
        self,
        dataset_path: str | Path,
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        subsample: int = 1,
        in_memory: bool = True,
    ):
        super().__init__()
        self.subsample = subsample

        self.dataset_path = Path(dataset_path)
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.in_memory = in_memory
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),  # force all images to 256x256
            ]
        )

        # ------------------------------------------------------------------
        # 1. Gather all pickle files
        # ------------------------------------------------------------------
        self.trajectories = []
        self.file_index = sorted(glob.glob(str(self.dataset_path / "*.pkl")))
        if not self.file_index:
            raise FileNotFoundError(f"No *.pkl files in {self.dataset_path}")

        self.trajectories = [None] * len(self.file_index)  # placeholder

        # ------------------------------------------------------------------
        # 3. Build a global index mapping (traj_id, start_step) → sample
        # ------------------------------------------------------------------
        # A valid *raw* window must span the following number of env steps:
        #     span = (obs_horizon + action_horizon + pred_horizon - 1) * subsample + 1
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

        # states: List[np.ndarray] = []
        # actions: List[np.ndarray] = []
        # for traj in (self._get_traj(i) for i in range(len(self.file_index))):
        #     states.extend([row["state"] for row in traj])
        #     actions.extend([row["action"] for row in traj])
        # states_np = np.asarray(states, dtype=np.float32)
        # actions_np = np.asarray(actions, dtype=np.float32)

        # self.stats = {
        #     "state": {
        #         "min": torch.from_numpy(states_np.min(axis=0)),
        #         "max": torch.from_numpy(states_np.max(axis=0)),
        #     },
        #     "action": {
        #         "min": torch.from_numpy(actions_np.min(axis=0)),
        #         "max": torch.from_numpy(actions_np.max(axis=0)),
        #     },
        # }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_traj(self, idx: int) -> List[dict]:
        """Return trajectory `idx` (load from disk if necessary)."""
        if self.trajectories[idx] is None:  # lazy‑load
            with open(self.file_index[idx], "rb") as f:
                self.trajectories[idx] = pickle.load(f)
        return self.trajectories[idx]  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        traj_id, start = self.sample_index[idx]
        traj = self._get_traj(traj_id)
        step = self.subsample

        # Build index lists -------------------------------------------------
        obs_indices = [start + i * step for i in range(self.obs_horizon)]
        pred_indices = [
            start + (self.obs_horizon + i) * step for i in range(self.pred_horizon)
        ]

        obs_seq = torch.tensor(
            [traj[t]["state"][6] for t in obs_indices], dtype=torch.float32
        )

        img_tensors = [self.transform(_image_to_tensor(traj[t]["state"][9])) for t in obs_indices]
        img_seq = torch.stack(img_tensors, dim=0)
        prediction_actions = torch.tensor(
            [traj[t]["action"] for t in pred_indices], dtype=torch.float32
        )

        return {
            "state": obs_seq,  # (obs_horizon, state_dim)
            "image": img_seq,  # (obs_horizon, im_dim)
            "action": prediction_actions,  # (pred_horizon, action_dim)
        }
