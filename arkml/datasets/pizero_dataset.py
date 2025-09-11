import torch
from PIL import Image
from torchvision import transforms

from arkml.core.dataset import ArkDataset
from arkml.core.registry import DATASETS


@DATASETS.register("pizero_dataset")
class PiZeroDataset(ArkDataset):

    def __init__(self, dataset_path, task_prompt, transform=None):
        self.dataset_path = dataset_path
        self.task_prompt = task_prompt
        self.transform = transform
        self.trajectories = []
        self._load_trajectories()

    def __getitem__(self, idx):
        trajectory = self.trajectories[idx]
        image = trajectory['state'][10]

        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        obs = {
            "image": image,  # (batch_size, C, H, W)
            "state": torch.tensor(trajectory['state'][:10], dtype=torch.float),  # (batch_size, state_dim)
            "action": torch.tensor(trajectory['action'], dtype=torch.float).unsqueeze(0),  # (batch_size, T, action_dim)
            "task": self.task_prompt
        }

        return obs
