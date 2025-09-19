import os
import pickle
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config_utils import resolve_visual_feature_names


class PiZeroDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        transform=None,
        *args,
        visual_input_features=None,
        image_base_index: int = 9,
        **kwargs,
    ):
        self.task_prompt = kwargs.pop("task_prompt", None)
        if self.task_prompt is None:
            raise ValueError("Missing required keyword 'task_prompt'")
        self.pred_horizon = 1

        super().__init__()
        self.dataset_path = dataset_path
        self.transform = transform or transforms.ToTensor()
        self.visual_input_features = resolve_visual_feature_names(
            visual_input_features
        )
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

        sample: dict[str, Any] = {"task": self.task_prompt}

        state_array = np.asarray(trajectory["state"][6], dtype=np.float32)
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
            sample[cam_name] = self._image_to_tensor(image_value)

        action_array = np.asarray(trajectory["action"], dtype=np.float32)
        if action_array.ndim == 1:
            action_array = action_array[None, :]
        sample["action"] = torch.from_numpy(action_array)

        return sample

    def _image_to_tensor(self, image_value: Any) -> torch.Tensor:
        array = np.asarray(image_value)
        if array.dtype != np.uint8:
            array_float = array.astype(np.float32)
            if array_float.max() <= 1.0:
                array_uint8 = np.clip(array_float * 255.0, 0, 255).astype(np.uint8)
            else:
                array_uint8 = np.clip(array_float, 0, 255).astype(np.uint8)
        else:
            array_uint8 = array

        image = Image.fromarray(array_uint8)
        return self.transform(image)
