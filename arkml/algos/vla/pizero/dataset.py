import os
import pickle
from typing import Any

import numpy as np
import torch
from arkml.utils.utils import _image_to_tensor
from torch.utils.data import Dataset
from torchvision import transforms


class PiZeroDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        transform=None,
        visual_input_features=None,
        image_base_index: int = 9,
        *args,
        **kwargs,
    ):
        self.pred_horizon = 1

        super().__init__()
        self.dataset_path = dataset_path
        self.transform = transform or transforms.ToTensor()
        self.visual_input_features = visual_input_features
        self.image_base_index = image_base_index

        self.index_map = []
        self._build_index_map()

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

        for fpath in file_list:
            with open(fpath, "rb") as f:
                traj_list = pickle.load(f)
                for idx in range(len(traj_list)):
                    self.index_map.append((fpath, idx))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx) -> dict[str, Any]:
        fpath, traj_idx = self.index_map[idx]
        with open(fpath, "rb") as f:
            traj_list = pickle.load(f)
            trajectory = traj_list[traj_idx]

        sample: dict[str, Any] = {"task": trajectory.get("prompt")}

        state_array = np.asarray(
            trajectory["state"][6], dtype=np.float32
        )  # TODO handle proper index based on data collection pipeline
        sample["state"] = torch.from_numpy(state_array)

        for cam_index, cam_name in enumerate(self.visual_input_features):
            image_value = trajectory.get(cam_name)
            if image_value is None:
                state_block = trajectory.get("state")
                if state_block is not None:
                    candidate_idx = self.image_base_index + cam_index
                    if len(state_block) > candidate_idx:
                        image_value = state_block[candidate_idx]
            if image_value is None:
                raise KeyError(f"Image data for '{cam_name}' not found in trajectory")
            image_t = _image_to_tensor(image_value)
            sample[cam_name] = self.transform(image_t)

        action_array = np.asarray(trajectory["action"], dtype=np.float32)
        if action_array.ndim == 1:
            action_array = action_array[None, :]
        sample["action"] = torch.from_numpy(action_array)

        return sample
