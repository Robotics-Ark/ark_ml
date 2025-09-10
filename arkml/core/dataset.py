import os
import pickle
from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class ArkDataset(Dataset, ABC):
    """
    Base abstract dataset class for loading trajectories of robot/environment data.

    This class provides the common functionality to load trajectory data from
    a directory of serialized files (e.g., .pkl files). Subclasses must implement
    the __getitem__ method to define how individual samples are returned.

    Attributes:
        dataset_path (str): Path to the directory containing dataset files.
        trajectories (List[Any]): List of loaded trajectory data.
    """

    def __init__(self, dataset_path: str):
        """
        Args:
            dataset_path (str): Path to the dataset directory containing trajectory files.
        """
        self.dataset_path: str = dataset_path
        self.trajectories: list[...] = []

    def _load_trajectories(self) -> None:
        """
        Loads all trajectories from the dataset directory into memory.

        This function assumes that the dataset directory contains files ending
        with ".pkl", where each file is a pickled list of trajectories.

        Raises:
            FileNotFoundError: If the dataset path does not exist.
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path '{self.dataset_path}' does not exist.")

        file_list = sorted(
            [os.path.join(self.dataset_path, f) for f in os.listdir(self.dataset_path) if f.endswith(".pkl")]
        )

        for fpath in file_list:
            with open(fpath, "rb") as f:
                traj_list = pickle.load(f)
                self.trajectories.append(traj_list)

    def __len__(self) -> int:
        """
        Returns:
            int: The number of trajectories loaded in the dataset.
        """
        return len(self.trajectories)

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, ...]:
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
        raise NotImplementedError
