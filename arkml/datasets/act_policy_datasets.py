from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from arkml.core.registry import DATASETS
from torchvision import transforms as T
from arkml.core.dataset import ArkDataset

tfm = T.Compose([
    T.ToTensor(),
    T.Resize((256, 256), antialias=True),
])

def _split_state(state: Any) -> Tuple[torch.Tensor, Any]:
    if not hasattr(state, "__getitem__"):
        raise TypeError(f"state must be indexable, got {type(state)}")
    if len(state) < 11:
        raise ValueError(f"state must have at least 11 items (10-dim + image), got {len(state)}")
    s_vec = state[:10]
    s_t = torch.as_tensor(s_vec, dtype=torch.float32)
    img = tfm(state[10])
    return s_t, img

def _to_action_tensor(action: Any) -> torch.Tensor:
    a = torch.as_tensor(action, dtype=torch.float32)
    if a.numel() != 8:
        raise ValueError(f"action must have 8 elements, got shape {tuple(a.shape)} with {a.numel()} elems")
    return a.reshape(8)

@DATASETS.register("act_dataset")
class ActionChunkingDataset(ArkDataset):
    """
    Returns samples of the form:
        - state: (10,)
        - image: as-is (index 10 of state list)
        - action_chunk: (K, 8)
        - action_mask: (K,)
        - next_state: (10,)
        - next_image: as-is

    Assumptions:
      - Each *trajectory* is a list of dict rows with keys: 'state', 'action', 'next_state'.
      - The dataset directory contains .pkl files. Each file may contain:
          (a) a single trajectory (list[dict]), or
          (b) a list of trajectories (list[list[dict]]).
    """

    def __init__(self, dataset_path: str, transform, chunk_size: int = 8 ):
        super().__init__(dataset_path=dataset_path)
        self.chunk_size = int(chunk_size)

        self._load_trajectories()

        # Normalize into a flat list of trajectories: List[List[dict]]
        self._trajs: List[List[dict]] = self._flatten_trajectories(self.trajectories)

        if len(self._trajs) == 0:
             raise FileNotFoundError(f"No trajectories found in '{dataset_path}'")
        self.transform = transform

        # Build flat index over (traj_idx, t0, L)
        self._index: List[Tuple[int, int, int]] = []
        self._build_index()

    def _flatten_trajectories(self, raw: List[Any]) -> List[List[dict]]:
        """
        Accepts:
          - raw[i] is either a trajectory (list[dict]) or a list of trajectories (list[list[dict]]).
        Produces:
          - flat list of trajectories (each trajectory is list[dict]).
        """
        flat: List[List[dict]] = []
        for item in raw:
            if isinstance(item, list) and len(item) > 0 and isinstance(item[0], dict):

                flat.append(item)
            elif isinstance(item, list) and len(item) > 0 and isinstance(item[0], list):

                for traj in item:
                    if not (isinstance(traj, list) and (len(traj) == 0 or isinstance(traj[0], dict))):
                        raise ValueError("Invalid trajectory structure inside file.")
                    flat.append(traj)
            elif isinstance(item, list) and len(item) == 0:

                continue
            else:
                raise ValueError("Each .pkl must contain a trajectory (list[dict]) or list of trajectories.")
        return flat

    def _build_index(self) -> None:
        self._index.clear()
        for ti, traj in enumerate(self._trajs):
            L = len(traj)
            for t0 in range(L):
                self._index.append((ti, t0, L))

    def __len__(self) -> int:
        # number of (trajectory, start) samples
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        traj_idx, t0, L = self._index[idx]
        traj = self._trajs[traj_idx]

        row0 = traj[t0]
        if "state" not in row0:
            raise KeyError(f"Row missing 'state' at traj={traj_idx} t={t0}")

        s_t, img_t = _split_state(row0["state"])

        K = self.chunk_size
        valid_len = min(K, L - t0)

        actions = torch.zeros((K, 8), dtype=torch.float32)
        mask = torch.zeros((K,), dtype=torch.float32)

        for k in range(valid_len):
            row = traj[t0 + k]
            if "action" not in row:
                raise KeyError(f"Row missing 'action' at traj={traj_idx} t={t0+k}")
            actions[k] = _to_action_tensor(row["action"])
            mask[k] = 1.0


        end_idx = min(t0 + K, L - 1)
        row_end = traj[end_idx]
        if "next_state" not in row_end:
            raise KeyError(f"Row missing 'next_state' at traj={traj_idx} t={end_idx}")

        s_next, img_next = _split_state(row_end["next_state"])

        return {
            "state": s_t,             # (10,)
            "image": img_t,           # original image object
            "action_chunk": actions,  # (K, 8)
            "action_mask": mask,      # (K,)
            "next_state": s_next,     # (10,)
            "next_image": img_next,   # original image object
            "meta": {
                "traj_index": traj_idx,
                "t0": t0,
                "traj_len": L,
                "effective_chunk": int(valid_len),
            },
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        states = torch.stack([b["state"] for b in batch], dim=0)
        actions = torch.stack([b["action_chunk"] for b in batch], dim=0)
        masks = torch.stack([b["action_mask"] for b in batch], dim=0)
        next_states = torch.stack([b["next_state"] for b in batch], dim=0)

        images = torch.stack([b["image"] for b in batch])
        next_images = torch.stack([b["next_image"] for b in batch])
        meta = [b["meta"] for b in batch]

        return {
            "state": states,
            "image": images,
            "action_chunk": actions,
            "action_mask": masks,
            "next_state": next_states,
            "next_image": next_images,
            "meta": meta,
        }
