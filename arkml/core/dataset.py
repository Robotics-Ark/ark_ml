from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import Dataset


class ArkDataset(Dataset, ABC):
    """
    Base abstract dataset class for loading trajectories of robot/environment data.

    This class provides the common functionality to load trajectory data from
    a directory of serialized files (e.g., .pkl files). Subclasses must implement
    the __getitem__ method to define how individual samples are returned.
    Attributes:
        dataset_path: Path to the directory containing dataset files.
    """

    def __init__(self, dataset_path: str, *args, **kwargs):
        self.dataset_path: str = dataset_path

    @abstractmethod
    def _build_index_map(self) -> None:
        """Build a lightweight index to enable lazy sample loading.

        Scans the dataset storage (e.g., directories, shard files, LMDB/HDF5/ZIP)
        and records, for each sample, only the minimal information required to
        retrieve it later without loading payloads (e.g., file path, shard/key,
        record index, byte offset/size, and optional metadata).
        This index allows
        `__len__` and `__getitem__` to be fast and memory‑efficient by deferring
        actual I/O until access time.

        Notes:
            - Keep entries small (no tensors/arrays) to minimize RAM usage.
            - Ensure deterministic ordering for reproducible splits.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """
        Provides the length of the dataset.

        Returns:
            int: The number of trajectories loaded in the dataset.
        """
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        """
        Abstract method to fetch a single sample from the dataset.

        Subclasses must implement this method to define how individual
        samples are returned.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            Dict[str, Any]: A dictionary representing the sample, which may
            include keys like 'state', 'image', 'action', etc.
        """
        ...
