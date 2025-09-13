import os
import pickle
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from arkml.core.dataset import ArkDataset


class PiZeroDataset(ArkDataset):
    """
    Lazy-loading dataset for PiZero trajectories stored as pickled files.
    Scans a directory of ``.pkl`` files and builds an in-memory index mapping each
    dataset index to a tuple ``(file_path, trajectory_index)``. Actual trajectory
    payloads are loaded on demand in ``__getitem__`` to keep memory usage low.

    Each trajectory is expected to contain keys like ``"state"`` and ``"action"``.
    This dataset returns a dictionary with image, state, action, and a task prompt
    suitable for the PiZero policy.

    Args:
        dataset_path: Directory containing pickled trajectory files (``.pkl``).
        transform: Optional image transform applied to the
            PIL image constructed from the trajectory state. If ``None``,
            a default ``transforms.ToTensor()`` is used.

    """

    def __init__(self, dataset_path, transform=None, *args, **kwargs):
        self.task_prompt = kwargs.pop("task_prompt", None)
        if self.task_prompt is None:
            raise ValueError("Missing required keyword 'task_prompt'")

        super().__init__(dataset_path)
        self.dataset_path = dataset_path
        self.transform = transform or transforms.ToTensor()

        # Store a list of (file_path, trajectory_index) to locate trajectories lazily
        self.index_map = []
        self._build_index_map()

    def _build_index_map(self) -> None:
        """Build an index map instead of loading all trajectories into memory."""
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

        for fpath in file_list:
            with open(fpath, "rb") as f:
                traj_list = pickle.load(f)
                for idx in range(len(traj_list)):
                    self.index_map.append((fpath, idx))

    def __len__(self) -> int:
        """Return the number of available trajectory entries.

        Returns:
        """
        return len(self.index_map)

    def __getitem__(self, idx) -> dict[str, Any]:
        """Load and return a single sample by dataset index.

        Uses the index map to find the corresponding (file, trajectory_idx),
        loads that trajectory lazily, and constructs the observation dictionary.

        The returned observation has:
            - "image": Image (C, H, W) from  `trajectory['state'][10]`,
            - "state": State (10, state_dim) from `trajectory['state'][:10]`.
            - "action": Action (1, action_dim) created from `trajectory['action']`.
            - "task": The task prompt provided at initialization.

        Args:
            idx (int): Dataset index.

        Returns:
            A dictionary with keys ``image``, ``state``, ``action``, and ``task``.
                  Tensors are placed on CPU; device management is the model's responsibility.

        """

        fpath, traj_idx = self.index_map[idx]

        # Load only the required trajectory
        with open(fpath, "rb") as f:
            traj_list = pickle.load(f)
            trajectory = traj_list[traj_idx]

        image = Image.fromarray(trajectory["state"][10])
        image = self.transform(image)

        obs = {
            "image": image,
            "state": torch.tensor(trajectory["state"][:10], dtype=torch.float),
            "action": torch.tensor(trajectory["action"], dtype=torch.float).unsqueeze(
                0
            ),
            "task": self.task_prompt,
        }

        return obs
