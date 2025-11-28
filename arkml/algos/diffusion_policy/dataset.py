import io
import pickle
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from arkml.core.app_context import ArkMLContext
from arkml.utils.utils import _image_to_tensor
from torch.utils.data import Dataset
from torchvision import transforms


class DiffusionPolicyDataset(Dataset):
    """Diffusion policy dataset"""

    def __init__(
        self,
        dataset_path: str | Path,
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        subsample: int = 1,
        transform=None,
        parquet_path: str | Path | None = None,
    ):
        """

        Args:
            dataset_path: Path to the directory containing trajectory files.
            pred_horizon: Number of future action steps to predict.
            obs_horizon: Number of observation frames used as context.
            action_horizon: Number of past actions to include.
            subsample: Step size for temporal subsampling.
            transform: Image transform.
            parquet_path: Path to the processed Parquet file.
        """
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.parquet_path = (
            Path(parquet_path)
            if parquet_path is not None
            else self.dataset_path / "processed.parquet"
        )
        self.pred_horizon = int(pred_horizon)
        self.obs_horizon = int(obs_horizon)
        self.action_horizon = int(action_horizon)
        self.subsample = max(1, int(subsample))
        self.transform = transform or transforms.Compose(
            [transforms.Resize((256, 256))]
        )

        # Ensure Parquet exists, then load it
        self._ensure_parquet()
        table = pq.read_table(self.parquet_path, memory_map=True)
        self.trajectories = self._table_to_trajectories(table)

        span = (
            self.obs_horizon + self.action_horizon + self.pred_horizon - 1
        ) * self.subsample + 1

        self.sample_index: list[tuple[int, int]] = []
        for tid, traj in enumerate(self.trajectories):
            t_len = traj["length"]
            if t_len < span:
                continue
            last_valid = t_len - span
            self.sample_index.extend([(tid, s) for s in range(last_valid + 1)])

    def _ensure_parquet(self) -> None:
        """
        Ensure that a processed Parquet file exists.
        Returns:
            None
        """
        if self.parquet_path.exists():
            return

        pkls = sorted(self.dataset_path.glob("*.pkl"))
        if not pkls:
            raise FileNotFoundError(f"No *.pkl files in {self.dataset_path}")

        def encode_image(arr: np.ndarray) -> bytes:
            img = Image.fromarray(arr.astype(np.uint8))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

        rows = []
        for pkl_path in pkls:
            with open(pkl_path, "rb") as f:
                traj: list[dict] = pickle.load(f)
            actions = [np.asarray(step["action"], dtype=np.float32) for step in traj]
            states = [
                np.asarray(step["state"][6][:8], dtype=np.float32) for step in traj
            ]
            images = [encode_image(np.asarray(step["state"][9])) for step in traj]
            rows.append(
                {
                    "traj_id": pkl_path.stem,
                    "actions": actions,
                    "state_vec": states,
                    "images": images,
                    "length": len(actions),
                }
            )

        schema = pa.schema(
            [
                ("traj_id", pa.string()),
                ("actions", pa.list_(pa.list_(pa.float32()))),
                ("state_vec", pa.list_(pa.list_(pa.float32()))),
                ("images", pa.list_(pa.binary())),
                ("length", pa.int32()),
            ]
        )
        batch = pa.Table.from_pylist(rows, schema=schema)
        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(batch, self.parquet_path, compression="zstd")

    @staticmethod
    def _table_to_trajectories(table: pa.Table) -> list[dict]:
        """
        Convert a Parquet table into a list of trajectory dictionaries.
        Args:
            table: A PyArrow table containing columns actions, state_vec, images, length

        Returns:
            A list of trajectories, each containing actions, state_vec, images, length

        """
        actions_col = table["actions"].to_pylist()
        states_col = table["state_vec"].to_pylist()
        images_col = table["images"].to_pylist()
        lengths = table["length"].to_pylist()

        trajectories: list[dict] = []
        for acts, sts, imgs, ln in zip(actions_col, states_col, images_col, lengths):
            traj = {
                "actions": np.asarray(acts, dtype=np.float32),
                "state_vec": np.asarray(sts, dtype=np.float32),
                "images": imgs,  # list of PNG bytes
                "length": int(ln),
            }
            trajectories.append(traj)
        return trajectories

    def __len__(self) -> int:
        """
        Return the number of available dataset samples.
        Returns:
            Total number of samples.
        """
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Return a single training sample at the given index.
        Args:
            idx: Index into the sample.

        Returns:
            A dictionary containing state,image, past_actions and future action.
        """
        traj_id, start = self.sample_index[idx]
        traj = self.trajectories[traj_id]
        step = self.subsample

        obs_indices = [start + i * step for i in range(self.obs_horizon)]
        past_indices = [
            start + self.obs_horizon - j * step - 1
            for j in reversed(range(self.action_horizon))
        ]
        pred_indices = [
            start + (self.obs_horizon + i) * step for i in range(self.pred_horizon)
        ]

        last_t = obs_indices[-1]
        latest_state = torch.from_numpy(traj["state_vec"][last_t]).float()

        image_bytes = traj["images"][last_t]
        image_arr = np.asarray(Image.open(io.BytesIO(image_bytes)))
        latest_image = self.transform(_image_to_tensor(image_arr))

        past_np = traj["actions"][past_indices]
        pred_np = traj["actions"][pred_indices]

        past_actions = torch.from_numpy(past_np)  # (action_horizon, action_dim)
        pred_actions = torch.from_numpy(pred_np)  # (pred_horizon, action_dim)

        return {
            "state": latest_state,  # (state_dim,)
            ArkMLContext.visual_input_features[0]: latest_image,  # (C,H,W)
            "past_actions": past_actions,  # (action_horizon, action_dim)
            "action": pred_actions,  # (pred_horizon, action_dim)
        }
