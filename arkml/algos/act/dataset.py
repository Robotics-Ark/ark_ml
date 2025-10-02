import glob
import pickle
from typing import Any, Optional

import torch
from torchvision import transforms as T
from torch.utils.data import Dataset

tfm = T.Compose(
    [
        T.ToTensor(),
        T.Resize((256, 256), antialias=True),
    ]
)


def _split_state(state: Any) -> tuple[torch.Tensor, Any]:
    """Extract (state_vector, image_tensor) from a raw state record.


    Expects an indexable ``state`` where the joint/state vector is at index 6 and
    the image-like object is at index 9.


    Args:
    state: Indexable record (e.g., list/tuple) holding sensors and image.


    Returns:
    Tuple[torch.Tensor, torch.Tensor]: ``(state_vec, image_tensor)``.


    Raises:
    TypeError: If ``state`` is not indexable.
    """

    if not hasattr(state, "__getitem__"):
        raise TypeError(f"state must be indexable, got {type(state)}")
    s_vec = state[6]
    s_t = torch.as_tensor(s_vec, dtype=torch.float32)
    img = tfm(state[9])
    return s_t, img


def _to_action_tensor(action: Any) -> torch.Tensor:
    """Convert an action to a 1D float32 tensor of length 8.

    Args:
    action: Array-like action.


    Returns:
    torch.Tensor: Shape ``(8,)``.


    Raises:
    ValueError: If action does not have exactly 8 elements.
    """
    a = torch.as_tensor(action, dtype=torch.float32)
    if a.numel() != 8:
        raise ValueError(
            f"action must have 8 elements, got shape {tuple(a.shape)} with {a.numel()} elems"
        )
    return a.reshape(8)


class ActionChunkingArkDataset(Dataset):
    """
    Trajectory dataset that yields fixed-length action chunks with masks.

    Sample format:
        - state: (10,)
        - image: as-is (index 10 of state list)
        - action_chunk: (K, 8)
        - action_mask: (K,)
        - next_state: (10,)
        - next_image: as-is
        - meta: {path, t0, traj_len, effective_chunk}

    Storage:
        Directory of .pkl trajectory files where each file is a list[dict] and each
        dict has keys: 'state', 'action', 'next_state' (and optionally others).

    Notes:
        - Lazy loads trajectories with a small LRU cache.
        - Index map stores (path, t0, L) for efficient random access.
    """

    def __init__(
        self,
        dataset_path: str,
        files: Optional[list[str]] = None,
        transform: T.Compose = None,
        chunk_size: int = 100,
    ):
        self.transform = transform
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.cache_size = 5
        self.dataset_path=dataset_path

        if files is None:
            self.files: list[str] = sorted(
                glob.glob(f"{self.dataset_path}/*.pkl")
            )
        else:
            self.files = sorted(files)

       # self.files = self.files[50:150]

        if not self.files:
            raise FileNotFoundError(
                f"No .pkl trajectories found in {self.dataset_path}"
            )

        # Simple LRU cache for trajectories
        self._cache: dict[str, list[dict]] = {}
        self._cache_order: List[str] = []

        # Lightweight index: list of (path, t0, traj_len)
        self._index: list[tuple[str, int, int]] = []
        self._build_index_map()

    # -------- Internal helpers --------
    def _get_traj(self, path: str) -> list[dict]:
        """Load (or fetch from cache) a trajectory list of dicts from disk."""
        if path in self._cache:
            # Refresh LRU
            if path in self._cache_order:
                self._cache_order.remove(path)
            self._cache_order.append(path)
            return self._cache[path]

        with open(path, "rb") as f:
            traj = pickle.load(f)
        if not isinstance(traj, list):
            raise ValueError(f"Trajectory {path} must be list, got {type(traj)}")

        self._cache[path] = traj
        self._cache_order.append(path)

        # Enforce LRU size
        while len(self._cache_order) > self.cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        return traj

    # -------- ArkDataset API --------
    def _build_index_map(self) -> None:
        """Build (path, t0, length) entries for each step in each trajectory."""
        self._index.clear()
        for path in self.files:
            traj = self._get_traj(path)
            L = len(traj)
            # Index every starting position t0 in the trajectory
            for t0 in range(L):
                self._index.append((path, t0, L))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Parameters
        ----------
        idx : int
        Dataset index.


        Returns
        -------
        dict
        Dictionary with keys:
        - ``state`` (torch.Tensor): state vector.
        - ``image`` (torch.Tensor): associated image tensor.
        - ``action_chunk`` (torch.Tensor): action sequence of shape (K, 8).
        - ``action_mask`` (torch.Tensor): mask of shape (K,) with 1 for valid steps.
        - ``meta`` (dict): metadata including file path, start index, trajectory length,
        and effective chunk length.
        """
        path, t0, L = self._index[idx]
        traj = self._get_traj(path)

        row0 = traj[t0]
        s_t, img_t = _split_state(row0["state"])

        K = self.chunk_size
        valid_len = min(K, L - t0)

        # Pre-allocate tensors (float32 for compatibility with most models)
        actions = torch.zeros((K, 8), dtype=torch.float32)
        mask = torch.zeros((K,), dtype=torch.float32 )

        for k in range(valid_len):
            row = traj[t0 + k]
            actions[k] = _to_action_tensor(row["action"])
            mask[k] = 1.0

        return {
            "state": s_t,  # (10,)
            "image": img_t,  # original image object
            "action_chunk": actions,  # (K, 8)
            "action_mask": mask,  # (K,)
            "meta": {
                "path": path,
                "t0": t0,
                "traj_len": L,
                "effective_chunk": valid_len,
            },
        }
